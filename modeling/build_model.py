import sys
sys.path.insert(0, '../')

import numpy as np
import random
import cv2
import os.path as osp

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
        self.poseAlignOp = PoseAlign(template_file=osp.dirname(osp.abspath(__file__))+'/templates.json', 
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
        self._setInputs(batchimgs, batchkpts, batchmasks)
        self._calcNetInputs()
        self._calcAlignMatrixs()        
        output = self._forward()        
        # self.visualize(output)
        return output
    
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
        for i, (matrix, kpts) in enumerate(zip(self.inputMatrixs, self.batchkpts)):
            m1 = matrix    
            # transform gt_kpts to feature coordinates.
            kpts = translib.warpAffineKpts(kpts, m2.dot(m1))
            
            self.featAlignMatrixs[i] = np.zeros((len(kpts), 3, 3), dtype=np.float32)
            self.maskAlignMatrixs[i] = np.zeros((len(kpts), 3, 3), dtype=np.float32)
            if self.cat_skeleton:
                self.skeletonFeats[i] = np.zeros((len(kpts), 55, size_align, size_align), dtype=np.float32)
                
            for j, kpt in enumerate(kpts):    
                timers['2'].tic()
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
                
                self.featAlignMatrixs[i][j] = m3
                self.maskAlignMatrixs[i][j] = m4.dot(m3).dot(m2).dot(m1)
                
                if self.cat_skeleton:
                    # size_align (sigma=3, threshold=1) for size_align=64
                    self.skeletonFeats[i][j] = genSkeletons(translib.warpAffineKpts([kpt], m3), 
                                                              size_align, size_align, 
                                                              stride=1, sigma=3, threshold=1,
                                                              visdiff = True).transpose(2, 0, 1)
                
                
    def _forward(self):
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
        return MaskOutput
    
    def _visualizeOutput(self, netOutput):
        outdir = './vis/'
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
        
