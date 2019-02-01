import os
import sys
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data
    
from lib.averageMeter import AverageMeters
from lib.logger import colorlogger
from lib.timer import Timers
from lib.averageMeter import AverageMeters
from lib.torch_utils import adjust_learning_rate

from modeling.build_model import Pose2Seg
from datasets.CocoDatasetInfo import CocoDatasetInfo, annToMask

# #CUDA_VISIBLE_DEVICES=3 python train.py
# NAME = 'pose2seg_nocat_cuda3_factor1.0'
# #CUDA_VISIBLE_DEVICES=4 python train.py
# NAME = 'pose2seg_nocat_cuda4_factor1.1'
# #CUDA_VISIBLE_DEVICES=5 python train.py
# NAME = 'pose2seg_nocat_cuda5_factor1.2'
#CUDA_VISIBLE_DEVICES=6 python train.py
NAME = 'pose2seg_nocat_cuda6_factor1.3'

# Set `LOG_DIR` and `SNAPSHOT_DIR`
def setup_logdir():
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) 
    LOGDIR = os.path.join('logs', '%s_%s'%(NAME, timestamp))
    SNAPSHOTDIR = os.path.join('snapshot', '%s_%s'%(NAME, timestamp))
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    if not os.path.exists(SNAPSHOTDIR):
        os.makedirs(SNAPSHOTDIR)
    return LOGDIR, SNAPSHOTDIR
LOGDIR, SNAPSHOTDIR = setup_logdir()

# Set logging 
logger = colorlogger(log_dir=LOGDIR, log_name='train_logs.txt')

# Set Global Timer
timers = Timers()

# Set Global AverageMeter
averMeters = AverageMeters()
    
def train(model, dataloader, optimizer, epoch, iteration):
    # switch to train mode
    model.train()
    
    averMeters.clear()
    end = time.time()
    for i, inputs in enumerate(dataloader): 
        averMeters['data_time'].update(time.time() - end)
        iteration += 1
        
        lr = adjust_learning_rate(optimizer, iteration, BASE_LR=0.0002,
                         WARM_UP_FACTOR=1.0/3, WARM_UP_ITERS=1000,
                         STEPS=(0, 14150*15, 14150*20), GAMMA=0.1)    # 408000, 528000
        
        # forward
        outputs = model(**inputs)
        
        # loss
        loss = outputs
            
        # backward
        averMeters['loss'].update(loss.data.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        averMeters['batch_time'].update(time.time() - end)
        end = time.time()
        
        if i % 10 == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Lr: [{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.5f} ({loss.avg:.5f})\t'
                  .format(
                      epoch, i, len(dataloader), lr, 
                      batch_time=averMeters['batch_time'], data_time=averMeters['data_time'],
                      loss=averMeters['loss'])
                 )
        
        if i % 10000 == 0:  
            torch.save(model.state_dict(), os.path.join(SNAPSHOTDIR, '%d_%d.pkl'%(epoch,i)))
            torch.save(model.state_dict(), os.path.join(SNAPSHOTDIR, 'last.pkl'))
        
    return iteration

class Dataset():
    def __init__(self):
        ImageRoot = '/home/dalong/nas/data/coco2017/train2017'
        AnnoFile = '/home/dalong/nas/data/coco2017/annotations/postprocess/person_keypoints_train2017_pose2seg.json'
        self.datainfos = CocoDatasetInfo(ImageRoot, AnnoFile, onlyperson=True, loadimg=True)
    
    def __len__(self):
        return len(self.datainfos)
    
    def __getitem__(self, idx):
        rawdata = self.datainfos[idx]
        img = rawdata['data']
        image_id = rawdata['id']
        
        height, width = img.shape[0:2]
        gt_kpts = np.float32(rawdata['gt_keypoints']).transpose(0, 2, 1) # (N, 17, 3)
        gt_segms = rawdata['segms']
        gt_masks = np.array([annToMask(segm, height, width) for segm in gt_segms])
    
        return {'img': img, 'kpts': gt_kpts, 'masks': gt_masks}
        
    def collate_fn(self, batch):
        batchimgs = [data['img'] for data in batch]
        batchkpts = [data['kpts'] for data in batch]
        batchmasks = [data['masks'] for data in batch]
        return {'batchimgs': batchimgs, 'batchkpts': batchkpts, 'batchmasks':batchmasks}
        
def test(model, dataset='cocoVal'):
    from datasets.CocoDatasetInfo import CocoDatasetInfo, annToMask
    from pycocotools import mask as maskUtils
    from tqdm import tqdm
    
    if dataset == 'OCHuman':
        ImageRoot = '/home/dalong/nas/data/OCHuman/v4/images'
        AnnoFile = '/home/dalong/nas/data/OCHuman/v4/OCHuman_v2_val_range_0.00_1.00.json'
    elif dataset == 'cocoVal':
        ImageRoot = '/home/dalong/nas/data/coco2017/val2017'
        AnnoFile = '/home/dalong/nas/data/coco2017/annotations/postprocess/person_keypoints_val2017_pose2seg.json'
    datainfos = CocoDatasetInfo(ImageRoot, AnnoFile, onlyperson=True, loadimg=True)
    
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
    
    timers.print()
    cocoEval = do_eval_coco(imgIds, datainfos.COCO, results_segm, 'segm')
    logger.info('[POSE2SEG]          AP|.5|.75| S| M| L|    AR|.5|.75| S| M| L|')
    _str = '[segm_score] %s '%dataset
    for value in cocoEval.stats.tolist():
        _str += '%.3f '%value
    logger.info(_str)
    
if __name__=='__main__':
    logger.info('===========> loading model <===========')
    model = Pose2Seg().cuda()
    #model.init('./init/coco/best.pkl')
    model.train()
    
    logger.info('===========> loading data <===========')
    datasetTrain = Dataset()
    dataloaderTrain = torch.utils.data.DataLoader(datasetTrain, batch_size=4, shuffle=True,
                                                   num_workers=4, pin_memory=False,
                                                   collate_fn=datasetTrain.collate_fn)


    logger.info('===========> set optimizer <===========')
    ''' set your optimizer like this. Normally is Adam/SGD. '''
    #optimizer = torch.optim.SGD(model.parameters(), 0.0002, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(model.parameters(), 0.0002, weight_decay=0.0000)

    iteration = 0
    epoch = 0
    try:
        while iteration < 14150*25:
            logger.info('===========>   training    <===========')
            iteration = train(model, dataloaderTrain, optimizer, epoch, iteration)
            epoch += 1
            
            logger.info('===========>   testing    <===========')
            test(model, dataset='cocoVal')
            test(model, dataset='OCHuman')


    except (KeyboardInterrupt):
        logger.info('Save ckpt on exception ...')
        torch.save(model.state_dict(), os.path.join(SNAPSHOTDIR, 'interrupt_%d_%d.pkl'%(epoch,iteration)))
        logger.info('Save ckpt done.')
