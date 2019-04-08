import sys
sys.path.insert(0, '../')

import numpy as np
import random
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import lib.transforms as translib
from lib.timer import Timers

from modeling.resnet import resnet50FPN
from modeling.affine_align import affine_align_gpu
from modeling.seg_module import resnet10units
from modeling.core import PoseAlign
from modeling.skeleton_feat import genSkeletons

timers = Timers()

class Pose2Seg(nn.Module):
    def __init__(self):
        super(Pose2Seg, self).__init__()
        self.MAXINST = 8
        ## size origin ->(m1)-> input ->(m2)-> feature ->(m3)-> align ->(m4)-> output
        self.size_input = 512
        self.size_feat = 128
        self.size_align = 64
        self.size_output = 64
        self.cat_skeleton = True
        
        self.backbone = resnet50FPN(pretrained=True)
        if self.cat_skeleton:
            self.segnet = resnet10units(256 + 55)  
        else:
            self.segnet = resnet10units(256)  
        self.poseAlignOp = PoseAlign(template_file='/home/dalong/nas/CVPR2019/Pose2Seg/modeling/templates.json', 
                                     visualize=False, factor = 1.0)
        
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.mean = np.ones((self.size_input, self.size_input, 3)) * mean
        self.mean = torch.from_numpy(self.mean.transpose(2, 0, 1)).cuda(0).float()
        
        self.std = np.ones((self.size_input, self.size_input, 3)) * std
        self.std = torch.from_numpy(self.std.transpose(2, 0, 1)).cuda(0).float()
        self.visCount = 0
        
        pass
    
    def init(self, path):
        pretrained_dict = torch.load(path)
        model_dict = self.state_dict()
        pretrained_dict = {k.replace('pose2seg.seg_branch', 'segnet'): v for k, v in pretrained_dict.items() \
                           if 'num_batches_tracked' not in k}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
    
    def forward(self, batchimgs, batchkpts, batchmasks=None):
        timers['(cpu+gpu) total'].tic()
        
        self._setInputs(batchimgs, batchkpts, batchmasks)
        
        timers['(cpu)pre-process2'].tic()
        self._calcNetInputs()
        timers['(cpu)pre-process2'].toc()
        
        timers['(cpu)pre-process3'].tic()
        self._calcAlignMatrixs()
        timers['(cpu)pre-process3'].toc()
        
        output = self._forward()
        timers['(cpu+gpu) total'].toc()
        
        # self.visualize(output)
        
        return output
    
    def visualize(self, output):
        for i, (img, outs) in enumerate(zip(self.batchimgs, output)):
            for j, out in enumerate(outs):
                cv2.imwrite('%d.jpg'%i, img)
                cv2.imwrite('%d_res%d.jpg'%(i, j), out*255)
        
    def _setInputs(self, batchimgs, batchkpts, batchmasks=None):
        ## batchimgs: a list of array (H, W, 3)
        ## batchkpts: a list of array (m, 17, 3)
        ## batchmasks: a list of array (m, H, W)
        self.batchimgs = batchimgs 
        self.batchkpts = batchkpts
        self.batchmasks = batchmasks
        self.bz = len(self.batchimgs)
        
        ## sample
        if self.training:
            ids = [(i, j) for i, kpts in enumerate(batchkpts) for j in range(len(kpts))]
            if len(ids) > self.MAXINST:
                select_ids = random.sample(ids, self.MAXINST)
                indexs = [[] for _ in range(self.bz)]
                for id in select_ids:
                    indexs[id[0]].append(id[1])

                for i, (index, kpts) in enumerate(zip(indexs, self.batchkpts)):
                    self.batchkpts[i] = self.batchkpts[i][index]
                    self.batchmasks[i] = self.batchmasks[i][index]

        
    def _calcNetInputs(self):
        self.inputMatrixs = [translib.get_aug_matrix(img.shape[1], img.shape[0], 512, 512, 
                                                      angle_range=(-0., 0.),
                                                      scale_range=(1., 1.), 
                                                      trans_range=(-0., 0.))[0] \
                             for img in self.batchimgs]
        
        inputs = [cv2.warpAffine(img, matrix[0:2], (512, 512)) \
                  for img, matrix in zip(self.batchimgs, self.inputMatrixs)]
        
        if len(inputs) == 1:
            inputs = inputs[0][np.newaxis, ...]
        else:
            inputs = np.array(inputs)
        
        inputs = inputs[..., ::-1]
        inputs = inputs.transpose(0, 3, 1, 2)
        inputs = inputs.astype('float32')     
        
        self.inputs = inputs
        
            
    def _calcAlignMatrixs(self):
        ## 1. transform kpts to feature coordinates.
        ## 2. featAlignMatrixs (size feature -> size align) used by affine-align
        ## 3. maskAlignMatrixs (size origin -> size output) used by Reverse affine-align
        ## matrix: size origin ->(m1)-> input ->(m2)-> feature ->(m3(mAug))-> align ->(m4)-> output
        size_input = self.size_input
        size_feat = self.size_feat
        size_align = self.size_align
        size_output = self.size_output
        m2 = translib.stride_matrix(size_feat / size_input)
        m4 = translib.stride_matrix(size_output / size_align)
        
        self.featAlignMatrixs = [[] for _ in range(self.bz)]
        self.maskAlignMatrixs = [[] for _ in range(self.bz)]
        if self.cat_skeleton:
            self.skeletonFeats = [[] for _ in range(self.bz)]        
        for i, (matrix, kpts, masks) in enumerate(zip(self.inputMatrixs, self.batchkpts, self.batchmasks)):
            m1 = matrix    
            # transform gt_kpts to feature coordinates.
            kpts = translib.warpAffineKpts(kpts, m2.dot(m1))
            
            self.featAlignMatrixs[i] = np.zeros((len(kpts), 3, 3), dtype=np.float32)
            self.maskAlignMatrixs[i] = np.zeros((len(kpts), 3, 3), dtype=np.float32)
            if self.cat_skeleton:
                self.skeletonFeats[i] = np.zeros((len(kpts), 55, size_align, size_align), dtype=np.float32)
                
            for j, (kpt, mask) in enumerate(zip(kpts, masks)):    
                timers['2'].tic()
                ## best_align: {'category', 'template', 'matrix', 'score', 'history'}
                best_align = self.poseAlignOp.align(kpt, size_feat, size_feat, 
                                                    size_align, size_align, 
                                                    visualize=False, return_history=False)
                timers['2'].toc()
                
                ## aug
                if self.training:
                    mAug, _ = translib.get_aug_matrix(size_align, size_align, 
                                                      size_align, size_align, 
                                                      angle_range=(-30, 30), 
                                                      scale_range=(0.8, 1.2), 
                                                      trans_range=(-0.1, 0.1))
                    m3 = mAug.dot(best_align['matrix'])
                else:
                    m3 = best_align['matrix']
                
                self.featAlignMatrixs[i][j] = m3
                self.maskAlignMatrixs[i][j] = m4.dot(m3).dot(m2).dot(m1)
                
                timers['4'].tic()
                if self.cat_skeleton:
                    # size_align (sigma=3, threshold=1) for size_align=64
                    self.skeletonFeats[i][j] = genSkeletons(translib.warpAffineKpts([kpt], m3), 
                                                              size_align, size_align, 
                                                              stride=1, sigma=3, threshold=1,
                                                              visdiff = True).transpose(2, 0, 1)
                timers['4'].toc()
                
                
    def _forward(self):
        timers['(gpu)'].tic()
        #########################################################################################################
        ## If we use `pytorch` pretrained model, the input should be RGB, and normalized by the following code:
        ##      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        ##                                       std=[0.229, 0.224, 0.225])
        ## Note: input[channel] = (input[channel] - mean[channel]) / std[channel], input is (0,1), not (0,255)
        #########################################################################################################
        inputs = (torch.from_numpy(self.inputs).cuda(0) / 255.0 - self.mean) / self.std
        [p1, p2, p3, p4] = self.backbone(inputs)
        feature = p1
        
        alignHs = np.vstack(self.featAlignMatrixs)
        indexs = np.hstack([idx * np.ones(len(m),) for idx, m in enumerate(self.featAlignMatrixs)])
        
        rois = affine_align_gpu(feature, indexs, 
                                 (self.size_align, self.size_align), 
                                 alignHs)

        if self.cat_skeleton:
            skeletons = np.vstack(self.skeletonFeats)
            skeletons = torch.from_numpy(skeletons).float().cuda(0)
            rois = torch.cat((rois, skeletons), 1)
        
        netOutput = self.segnet(rois)
        
        
        if self.training:
            loss = self._calcLoss(netOutput)
            return loss
        else:
            netOutput = F.softmax(netOutput, 1)
            timers['(gpu)'].toc()
            netOutput = netOutput.detach().data.cpu().numpy()
            output = self._getMaskOutput(netOutput)
            
            if self.visCount < 0:
                self._visualizeOutput(netOutput)
                self.visCount += 1
            
            return output 
        
    def _calcLoss(self, netOutput):
        mask_loss_func = nn.CrossEntropyLoss(ignore_index=255)
        
        gts = []
        for masks, Matrixs in zip(self.batchmasks, self.maskAlignMatrixs):
            for mask, matrix in zip(masks, Matrixs):
                gts.append(cv2.warpAffine(mask, matrix[0:2], (self.size_output, self.size_output)))
        gts = torch.from_numpy(np.array(gts)).long().cuda(0)
        
        loss = mask_loss_func(netOutput, gts)
        return loss
        
        
    def _getMaskOutput(self, netOutput):
        timers['(cpu)_getMaskOutput'].tic()

        netOutput = netOutput.transpose(0, 2, 3, 1)        
        MaskOutput = [[] for _ in range(self.bz)]
        
        idx = 0
        for i, (img, kpts) in enumerate(zip(self.batchimgs, self.batchkpts)):
            height, width = img.shape[0:2]
            for j in range(len(kpts)):
                predmap = netOutput[idx]
                H_e2e = self.maskAlignMatrixs[i][j]
                
                pred_e2e = cv2.warpAffine(predmap, H_e2e[0:2], (width, height), 
                                          borderMode=cv2.BORDER_CONSTANT,
                                          flags=cv2.WARP_INVERSE_MAP+cv2.INTER_LINEAR) 
                               
                pred_e2e = pred_e2e[:, :, 1]
                pred_e2e[pred_e2e>0.5] = 1
                pred_e2e[pred_e2e<=0.5] = 0
                mask = pred_e2e.astype(np.uint8) 
                MaskOutput[i].append(mask)                
                
                idx += 1
        timers['(cpu)_getMaskOutput'].toc()
        return MaskOutput
    
    def _visualizeOutput(self, netOutput):
        outdir = '/home/dalong/nas/CVPR2019/Pose2Seg/vis/'
        netOutput = netOutput.transpose(0, 2, 3, 1)        
        MaskOutput = [[] for _ in range(self.bz)]
        
        mVis = translib.stride_matrix(4)
        
        idx = 0
        for i, (img, masks) in enumerate(zip(self.batchimgs, self.batchmasks)):
            height, width = img.shape[0:2]
            for j in range(len(masks)):
                predmap = netOutput[idx]
                
                predmap = predmap[:, :, 1]
                predmap[predmap>0.5] = 1
                predmap[predmap<=0.5] = 0
                predmap = cv2.cvtColor(predmap, cv2.COLOR_GRAY2BGR)
                predmap = cv2.warpAffine(predmap, mVis[0:2], (256, 256))
                
                matrix = self.maskAlignMatrixs[i][j]
                matrix = mVis.dot(matrix)
                
                imgRoi = cv2.warpAffine(img, matrix[0:2], (256, 256))
                
                mask = cv2.warpAffine(masks[j], matrix[0:2], (256, 256))
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                
                I = np.logical_and(mask, predmap)
                U = np.logical_or(mask, predmap)
                iou = I.sum() / U.sum()
                
                vis = np.hstack((imgRoi, mask*255, predmap*255))
                cv2.imwrite(outdir + '%d_%d_%.2f.jpg'%(self.visCount, j, iou), np.uint8(vis))
                
                idx += 1
        
def do_eval_coco(image_ids, coco, results, flag):
    from pycocotools.cocoeval import COCOeval
    assert flag in ['bbox', 'segm', 'keypoints']
    # Evaluate
    coco_results = coco.loadRes(results)
    cocoEval = COCOeval(coco, coco_results, flag)
    cocoEval.params.imgIds = image_ids
    cocoEval.params.catIds = [1]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize() 
    # print(cocoEval.ious)
    return cocoEval

        
if __name__ == '__main__':
    from datasets.CocoDatasetInfo import CocoDatasetInfo, annToMask
    from pycocotools import mask as maskUtils
    from tqdm import tqdm
    
#     ImageRoot = '/home/dalong/nas/data/coco2017/val2017' 
#     AnnoFile = '/home/dalong/nas/data/coco2017/annotations/postprocess/person_keypoints_val2017_pose2seg.json'
    ImageRoot = '/home/dalong/nas/data/OCHuman/v4/images' # 0.071
    AnnoFile = '/home/dalong/nas/data/OCHuman/v4/OCHuman_v2_all_range_0.00_1.00.json'
    datainfos = CocoDatasetInfo(ImageRoot, AnnoFile, onlyperson=True, loadimg=True)
    
    model = Pose2Seg().cuda(0)
    #model.init('../init/coco/best.pkl')
    model.init('nocat.pkl')
    model.eval()
    
    Ntotal = 0
    
    results_segm = []
    imgIds = []
    for i in tqdm(range(len(datainfos))):
        rawdata = datainfos[i]
        img = rawdata['data']
        image_id = rawdata['id']
        
        height, width = img.shape[0:2]
        gt_kpts = np.float32(rawdata['gt_keypoints']).transpose(0, 2, 1) # (N, 17, 3)
        gt_segms = rawdata['segms']
        gt_masks = np.array([annToMask(segm, height, width) for segm in gt_segms])
        Ntotal += len(gt_kpts)
        #print (height, width, len(gt_kpts))
        
        output = model([img], [gt_kpts], [gt_masks])
    
        
#         for mask in gt_masks:
#             cv2.imwrite('/home/dalong/nas/CVPR2019/Pose2Seg/vis/gt.jpg', np.uint8(mask*255))
    
        for mask in output[0]:
            #cv2.imwrite('/home/dalong/nas/CVPR2019/Pose2Seg/vis/out.jpg', np.uint8(mask*255))
            maskencode = maskUtils.encode(np.asfortranarray(mask))
            maskencode['counts'] = maskencode['counts'].decode('ascii')
            results_segm.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "score": 1.0,
                    "segmentation": maskencode
                })
        imgIds.append(image_id)
    
    timers.print()
    cocoEval = do_eval_coco(imgIds, datainfos.COCO, results_segm, 'segm')
    print('[POSE2SEG] AP|.5|.75| S| M| L|    AR|.5|.75| S| M| L|')
    _str = '[segm_score] '
    for value in cocoEval.stats.tolist():
        _str += '%.3f '%value
    print(_str)
    print('AvegPerson:', Ntotal/len(imgIds))
    
    