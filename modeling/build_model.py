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
        self.cat_skeleton = False
        
        self.backbone = resnet50FPN(pretrained=True)
        if self.cat_skeleton:
            self.segnet = resnet10units(256 + 55)  
        else:
            self.segnet = resnet10units(256)  
        self.poseAlignOp = PoseAlign(template_file='/home/dalong/nas/CVPR2019/Pose2Seg/modeling/templates.json', visualize=False)
        
        pass
    
    def init(self, path):
        pretrained_dict = torch.load(path)
        model_dict = self.state_dict()
        pretrained_dict = {k.replace('pose2seg.seg_branch', 'segnet'): v for k, v in pretrained_dict.items()}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
    
    def forward(self, batchimgs, batchkpts, batchmasks=None):
        timers['(cpu+gpu) total'].tic()
        timers['(cpu)pre-process'].tic()
        self._setInputs(batchimgs, batchkpts, batchmasks)
        self._calcNetInputs()
        self._calcAlignMatrixs()
        timers['(cpu)pre-process'].toc()
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
        inputs = preprocess(np.array(inputs)).transpose(0, 3, 1, 2).astype('float32')     
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
            
            for kpt, mask in zip(kpts, masks):    
                ## best_align: {'category', 'template', 'matrix', 'score', 'history'}
                best_align = self.poseAlignOp.align(kpt, size_feat, size_feat, 
                                                    size_align, size_align, 
                                                    visualize=False, return_history=False)
                
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
                
                self.featAlignMatrixs[i].append(m3)
                self.maskAlignMatrixs[i].append(m4.dot(m3).dot(m2).dot(m1))
                
                if self.cat_skeleton:
                    # size_align (sigma=3, threshold=1) for size_align=64
                    self.skeletonFeats[i].append(genSkeletons(translib.warpAffineKpts([kpt], m3), 
                                                              size_align, size_align, 
                                                              stride=1, sigma=3, threshold=1,
                                                              visdiff = True).transpose(2, 0, 1))
                
                
            self.featAlignMatrixs[i] = np.array(self.featAlignMatrixs[i]).reshape(-1, 3, 3)
            self.maskAlignMatrixs[i] = np.array(self.maskAlignMatrixs[i]).reshape(-1, 3, 3)
            if self.cat_skeleton:
                self.skeletonFeats[i] = np.array(self.skeletonFeats[i]).reshape(-1, 55, size_align, size_align)
            
    def _forward(self):
        inputs = torch.from_numpy(self.inputs).cuda(0)
        timers['(gpu)backbone'].tic()
        [p1, p2, p3, p4] = self.backbone(inputs)
        feature = p1
        timers['(gpu)backbone'].toc()
        
        alignHs = np.vstack(self.featAlignMatrixs)
        indexs = np.hstack([idx * np.ones(len(m),) for idx, m in enumerate(self.featAlignMatrixs)])
        
        timers['(gpu)affine_align_gpu'].tic()
        rois = affine_align_gpu(feature, indexs, 
                                 (self.size_align, self.size_align), 
                                 alignHs)

        if self.cat_skeleton:
            skeletons = np.vstack(self.skeletonFeats)
            skeletons = torch.from_numpy(skeletons).float().cuda(0)
            rois = torch.cat((rois, skeletons), 1)
        
        netOutput = self.segnet(rois)
        timers['(gpu)affine_align_gpu'].toc()
        
        if self.training:
            loss = self._calcLoss(netOutput)
            return loss
        else:
            output = self._getMaskOutput(netOutput)
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
        netOutput = netOutput.detach().data.cpu().numpy().transpose(0, 2, 3, 1)
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
            
                pred_e2e = F.softmax(torch.from_numpy(pred_e2e), 2)[:, :, 1] # H, W
                pred_e2e = pred_e2e.numpy()

                pred_e2e[pred_e2e>0.5] = 1
                pred_e2e[pred_e2e<=0.5] = 0
                mask = pred_e2e.astype(np.uint8) 
                MaskOutput[i].append(mask)
                
                idx += 1
        timers['(cpu)_getMaskOutput'].toc()
        return MaskOutput
     
def preprocess(img): 
    #########################################################################################################
    ## If we use `pytorch` pretrained model, the input should be RGB, and normalized by the following code:
    ##      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    ##                                       std=[0.229, 0.224, 0.225])
    ## Note: input[channel] = (input[channel] - mean[channel]) / std[channel], input is (0,1), not (0,255)
    #########################################################################################################
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return (np.float32(img[..., ::-1]) / 255.0 - mean) / std

        
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
    return cocoEval

        
if __name__ == '__main__':
    from datasets.CocoDatasetInfo import CocoDatasetInfo, annToMask
    from pycocotools import mask as maskUtils
    from tqdm import tqdm
    
    ImageRoot = '/home/dalong/nas/data/coco2017/val2017'
    AnnoFile = '/home/dalong/nas/data/coco2017/annotations/postprocess/person_keypoints_val2017_pose2seg.json'
    datainfos = CocoDatasetInfo(ImageRoot, AnnoFile, onlyperson=True, loadimg=True)
    
    model = Pose2Seg().cuda(0)
    model.init('../init/coco/best.pkl')
    model.eval()
    
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
    
        #print (height, width, len(gt_kpts))
        
        output = model([img], [gt_kpts], [gt_masks])
    
        for mask in output[0]:
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
    
    
    