##############################################################
# COCO dataset Info Loader. Clear unvalid items. 
##############################################################

import os
import numpy as np
import scipy.sparse
import cv2
import copy
import math
from pycocotools.coco import COCO
import pycocotools.mask as mask_util



def annToMask(segm, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    def _annToRLE(segm, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_util.frPyObjects(segm, height, width)
            rle = mask_util.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = mask_util.frPyObjects(segm, height, width)
        else:
            # rle
            rle = segm
        return rle

    rle = _annToRLE(segm, height, width)
    mask = mask_util.decode(rle)
    return mask

class CocoDatasetInfo():
    def __init__(self, ImageRoot, AnnoFile, onlyperson=False, loadimg=True):

        ''' **Just** loading coco dataset, with necessary pre-process:
        1. obj['segmentation'] polygons should have >= 3 points, so require >= 6 coordinates
        2. obj['area'] should >= GT_MIN_AREA
        3. ignore objs with obj['ignore']==1
        4. IOU(bbox, img) should > 0,  Area(bbox) should > 0

        Attributes: 
            self.category_to_id_map
            self.classes
            self.num_classes : 81
            self.json_category_id_to_contiguous_id
            self.contiguous_category_id_to_json_id
            self.image_ids : <class 'list'>

            self.keypoints
            self.keypoint_flip_map
            self.keypoints_to_id_map
            self.num_keypoints : 17

        Tools:
            rawdata = self.flip_rawdata(rawdata)
        '''
        self.GT_MIN_AREA = 0
        self.loadimg = loadimg

        self.imgroot = ImageRoot
        self.COCO = COCO(AnnoFile)
        
        # Set up dataset classes
        if onlyperson:
            self.category_ids = [1]
        else:
            self.category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(self.category_ids)]
        self.category_to_id_map = dict(zip(categories, self.category_ids))
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1
            for i, v in enumerate(self.category_ids)
        }
        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }
        
        # self.__len__() reference to self.image_ids
        self.image_ids = self.COCO.getImgIds(catIds=self.category_ids)
        self.image_ids.sort()
        #self.image_ids = self.image_ids[0:200] # for debug.
        #self.image_ids = [9,9,9,9,9]
        
        # Initialize COCO keypoint information.
        self.keypoints = None
        self.keypoint_flip_map = None
        self.keypoints_to_id_map = None
        self.num_keypoints = 0
        # Thus far only the 'person' category has keypoints
        if 'person' in self.category_to_id_map:
            cat_info = self.COCO.loadCats([self.category_to_id_map['person']])
            # Check if the annotations contain keypoint data or not
            if 'keypoints' in cat_info[0]:
                keypoints = cat_info[0]['keypoints']
                self.keypoints_to_id_map = dict(
                    zip(keypoints, range(len(keypoints))))
                self.keypoints = keypoints
                self.num_keypoints = len(keypoints)
                self.keypoint_flip_map = {
                    'left_eye': 'right_eye',
                    'left_ear': 'right_ear',
                    'left_shoulder': 'right_shoulder',
                    'left_elbow': 'right_elbow',
                    'left_wrist': 'right_wrist',
                    'left_hip': 'right_hip',
                    'left_knee': 'right_knee',
                    'left_ankle': 'right_ankle'}

        self.roidb = None # Pre-load

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        if self.roidb is None:
            return self.getitem(idx)
        else:
            return self.roidb[idx]

    def getitem(self, idx):
        ''' **Just** loading coco dataset, with necessary pre-process:
        1. obj['segmentation'] polygons should have >= 3 points, so require >= 6 coordinates
        2. obj['area'] should >= GT_MIN_AREA
        3. ignore objs with obj['ignore']==1
        4. IOU(bbox, img) should > 0,  Area(bbox) should > 0
        
        Return:
            rawdata {
                dataset': self,
                'id': image_id,
                'image': os.path.join(self.imgroot, datainfo['file_name']),
                'width': datainfo['width'],
                'height': datainfo['height'],
                
                'flipped': False,
                'has_visible_keypoints': False/True,
                'boxes': np.empty((GtN, 4), dtype=np.float32),
                'segms': [GtN,],
                'gt_classes': np.empty((GtN), dtype=np.int32),
                'seg_areas': np.empty((GtN), dtype=np.float32),
                'gt_overlaps': scipy.sparse.csr_matrix(
                                np.empty((GtN, 81), dtype=np.float32)
                                ),
                'is_crowd': np.empty((GtN), dtype=np.bool),
                'box_to_gt_ind_map': np.empty((GtN), dtype=np.int32) 

                if self.keypoints is not None:
                'gt_keypoints': np.empty((GtN, 3, self.num_keypoints), dtype=np.int32)

            }
        '''

        # ---------------------------
        # _prep_roidb_entry()    
        # ---------------------------
        image_id = self.image_ids[idx]
        datainfo = self.COCO.loadImgs(image_id)[0]
        rawdata = {
            #'dataset': self,
            #'flickr_url': datainfo['flickr_url'],
            'id': image_id,
            #'coco_url': datainfo['coco_url'],
            'image': os.path.join(self.imgroot, datainfo['file_name']),
            'data': cv2.imread(os.path.join(self.imgroot, datainfo['file_name'])) if self.loadimg else None,
            'width': datainfo['width'],
            'height': datainfo['height'],
            
            'flipped': False,
            'has_visible_keypoints': False,
            'boxes': np.empty((0, 4), dtype=np.float32),
            'segms': [],
            'gt_classes': np.empty((0), dtype=np.int32),
            'seg_areas': np.empty((0), dtype=np.float32),
            'gt_overlaps': scipy.sparse.csr_matrix(
                            np.empty((0, self.num_classes), dtype=np.float32)
                            ),
            'is_crowd': np.empty((0), dtype=np.bool),
            # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
            # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
            'box_to_gt_ind_map': np.empty((0), dtype=np.int32),
            # The only difference between gt_classes v.s. max_classes is about 'crowd' objs.
            'max_classes': np.empty((0), dtype=np.int32),
            'max_overlaps': np.empty((0), dtype=np.float32),
        }
        if self.keypoints is not None:
            rawdata['gt_keypoints'] = np.empty((0, 3, self.num_keypoints), dtype=np.int32)
            
        # ---------------------------
        # _add_gt_annotations()
        # ---------------------------
        # Include ground-truth object annotations
        """Add ground truth annotation metadata to an roidb entry."""
        ann_ids = self.COCO.getAnnIds(imgIds=rawdata['id'], catIds=self.category_ids, iscrowd=None)
        objs = self.COCO.loadAnns(ann_ids)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_segms = []
        width = rawdata['width']
        height = rawdata['height']
        for obj in objs:
            # crowd regions are RLE encoded and stored as dicts
            if isinstance(obj['segmentation'], list):
                # Valid polygons have >= 3 points, so require >= 6 coordinates
                obj['segmentation'] = [
                    p for p in obj['segmentation'] if len(p) >= 6
                ]
            if obj['area'] < self.GT_MIN_AREA:
                continue
            if 'ignore' in obj and obj['ignore'] == 1:
                continue
            # Convert form (x1, y1, w, h) to (x1, y1, x2, y2)
            x1, y1, bboxw, bboxh  = obj['bbox']
            x1, y1, x2, y2 = [x1, y1, x1 + bboxw - 1, y1 + bboxh - 1] # Note: -1 for h and w
            x1 = min(width - 1., max(0., x1))
            y1 = min(height - 1., max(0., y1))
            x2 = min(width - 1., max(0., x2))
            y2 = min(height - 1., max(0., y2))
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                valid_segms.append(obj['segmentation'])
        num_valid_objs = len(valid_objs)

        if num_valid_objs==0: ## is_valid
            # print ('ignore %d'%idx)
            return self.getitem(idx+1)

        boxes = np.zeros((num_valid_objs, 4), dtype=rawdata['boxes'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=rawdata['gt_classes'].dtype)
        gt_overlaps = np.zeros(
            (num_valid_objs, self.num_classes),
            dtype=rawdata['gt_overlaps'].dtype
        )
        seg_areas = np.zeros((num_valid_objs), dtype=rawdata['seg_areas'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=rawdata['is_crowd'].dtype)
        box_to_gt_ind_map = np.zeros(
            (num_valid_objs), dtype=rawdata['box_to_gt_ind_map'].dtype
        )
        if self.keypoints is not None:
            gt_keypoints = np.zeros(
                (num_valid_objs, 3, self.num_keypoints),
                dtype=rawdata['gt_keypoints'].dtype
            )

        im_has_visible_keypoints = False
        for ix, obj in enumerate(valid_objs):
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            is_crowd[ix] = obj['iscrowd']
            box_to_gt_ind_map[ix] = ix
            if self.keypoints is not None:
                gt_keypoints[ix, :, :] = self._get_gt_keypoints(obj)
                if np.sum(gt_keypoints[ix, 2, :]) > 0:
                    im_has_visible_keypoints = True
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                gt_overlaps[ix, :] = -1.0
            else:
                gt_overlaps[ix, cls] = 1.0
        rawdata['boxes'] = np.append(rawdata['boxes'], boxes, axis=0)
        rawdata['segms'].extend(valid_segms)
        # To match the original implementation:
        # rawdata['boxes'] = np.append(
        #     rawdata['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
        rawdata['gt_classes'] = np.append(rawdata['gt_classes'], gt_classes)
        rawdata['seg_areas'] = np.append(rawdata['seg_areas'], seg_areas)
        rawdata['gt_overlaps'] = np.append(
            rawdata['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        rawdata['gt_overlaps'] = scipy.sparse.csr_matrix(rawdata['gt_overlaps'])
        rawdata['is_crowd'] = np.append(rawdata['is_crowd'], is_crowd)
        rawdata['box_to_gt_ind_map'] = np.append(
            rawdata['box_to_gt_ind_map'], box_to_gt_ind_map
        )
        if self.keypoints is not None:
            rawdata['gt_keypoints'] = np.append(
                rawdata['gt_keypoints'], gt_keypoints, axis=0
            )
            rawdata['has_visible_keypoints'] = im_has_visible_keypoints


        '''
        The only difference between gt_classes v.s. max_classes is about 'crowd' objs.
        In max_classes, crowd objs are signed as bg.
                    bg    cls1    cls2    cls3   | gt_classes | max_overlaps |   max_classes
        obj1        0.0   1.0     0.0     0.0    | 1(cls1)    | 1.0          |   1(cls1)
        obj2(crowd) -1.0  -1.0    -1.0    -1.0   | 3(cls3)    | -1.0         |   0(bg)
        ibj3        0.0   0.0     1.0     0.0    | 2(cls2)    | 1.0          |   2(cls2)
        '''
        gt_overlaps = rawdata['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        rawdata['max_classes'] = max_classes
        rawdata['max_overlaps'] = max_overlaps

        return rawdata
    
    def _get_gt_keypoints(self, obj):
        """Return ground truth keypoints."""
        if 'keypoints' not in obj:
            return None
        kp = np.array(obj['keypoints'])
        x = kp[0::3]  # 0-indexed x coordinates
        y = kp[1::3]  # 0-indexed y coordinates
        # 0: not labeled; 1: labeled, not inside mask;
        # 2: labeled and inside mask
        v = kp[2::3]
        num_keypoints = len(obj['keypoints']) / 3
        assert num_keypoints == self.num_keypoints
        gt_kps = np.ones((3, self.num_keypoints), dtype=np.int32)
        for i in range(self.num_keypoints):
            gt_kps[0, i] = x[i]
            gt_kps[1, i] = y[i]
            gt_kps[2, i] = v[i]
        return gt_kps

    def transform_rawdata(self, rawdata, matrix, dstwidth, dstheight):
        '''
        See `get_affine_matrix` about the document of `matrix`.
        Note that the padding strategies for image and segms are both (0,0,0). I recomand you to sub MEAN before
        this operation. If you have other request, you should overwrite this function. (warning)
        size_related_keys = ['width', 'height', 'seg_areas', 'data', 'boxes', 'segms', 'gt_keypoints']
        '''
        assert matrix.shape == (2,3)
        # image
        rawdata['data'] = cv2.warpAffine(rawdata['data'], matrix, (dstwidth, dstheight), flags=cv2.INTER_LINEAR, 
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        GtN = len(rawdata['segms'])
        # segm
        for i in range(GtN):
            if isinstance(rawdata['segms'][i], dict):
                mask = annToMask(rawdata['segms'][i], rawdata['height'], rawdata['width'])
                mask = cv2.warpAffine(mask, matrix, (dstwidth, dstheight), flags=cv2.INTER_NEAREST, 
                                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0) # or 255
                rawdata['segms'][i] = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            elif isinstance(rawdata['segms'][i], list):
                for poly_id, poly in enumerate(rawdata['segms'][i]):
                    cors = np.array(poly).reshape(-1, 2)
                    cors_new = np.hstack((cors, np.ones((len(cors), 1), np.float32))).dot(matrix.T)
                    cors_new[:, 0] = np.clip(cors_new[:, 0], 0, dstwidth)
                    cors_new[:, 1] = np.clip(cors_new[:, 1], 0, dstheight)
                    rawdata['segms'][i][poly_id] = cors_new.flatten().tolist()[0]
            else:
                print ('segm type error!')
        # box: (GtN,2) -> (GtN,3)(dot)(3,2) -> (GtN,2)
        rawdata['boxes'][:, 0:2] = np.hstack((rawdata['boxes'][:, 0:2], np.ones((GtN, 1), np.float32))).dot(matrix.T)
        rawdata['boxes'][:, 2:4] = np.hstack((rawdata['boxes'][:, 2:4], np.ones((GtN, 1), np.float32))).dot(matrix.T)
        rawdata['boxes'][:, 0::2] = np.clip(rawdata['boxes'][:, 0::2], 0, dstwidth) # -1 ?
        rawdata['boxes'][:, 1::2] = np.clip(rawdata['boxes'][:, 1::2], 0, dstheight)
        if self.keypoints is not None:
            # gt_keypoint: (GtN,2,NumKpt) -> (GtN,NumKpt,3)(dot)(3,2) -> (GtN,NumKpt,2) -> (GtN,2,NumKpt)
            rawdata['gt_keypoints'][:, 0:2, :] = \
                    np.stack((rawdata['gt_keypoints'][:, 0:2, :].transpose((0, 2, 1)),
                               np.ones((GtN, self.num_keypoints, 1), np.float32)), axis=2).dot(matrix.T).transpose((0, 2, 1))
            inds = np.where(rawdata['gt_keypoints'][:, 2, :] == 0)
            rawdata['gt_keypoints'][inds[0], 0, inds[1]] = 0
            rawdata['gt_keypoints'][:, 0, :] = np.clip(rawdata['gt_keypoints'][:, 0, :], 0, dstwidth) # -1 ?
            rawdata['gt_keypoints'][:, 1, :] = np.clip(rawdata['gt_keypoints'][:, 0, :], 0, dstheight)
        # infos
        rawdata['width'] = dstwidth
        rawdata['height'] = dstheight
        rawdata['seg_areas'] = rawdata['seg_areas'] * math.sqrt(matrix[0,0]**2+matrix[1,0]**2) \
                                                    * math.sqrt(matrix[0,1]**2+matrix[1,1]**2)
        return rawdata
        
    def flip_rawdata_inplace(self, rawdata):
        # inplace flip
        rawdata['boxes'] = self.flip_boxes(rawdata['boxes'], rawdata['width'])
        rawdata['segms'] = self.flip_segms(
            rawdata['segms'], rawdata['height'], rawdata['width']
        )
        if self.keypoints is not None:
            rawdata['gt_keypoints'] = self.flip_keypoints(
                self.keypoints, self.keypoint_flip_map,
                rawdata['gt_keypoints'], rawdata['width']
            )
        rawdata['data'] = np.fliplr(rawdata['data'])
        rawdata['flipped'] = True
        return rawdata

    def flip_boxes(self, boxes, width):
        flipped_boxes = boxes.copy()
        flipped_boxes[:, 0] = width - boxes[:, 2] - 1 # Note: -1
        flipped_boxes[:, 2] = width - boxes[:, 0] - 1
        assert (flipped_boxes[:, 2] >= flipped_boxes[:, 0]).all()
        return flipped_boxes

    def flip_segms(self, segms, height, width): 
        """Left/right flip each mask in a list of masks."""
        flipped_segms = []
        for segm in segms:
            mask = np.fliplr(annToMask(segm, height, width))
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            flipped_segms.append(rle)
        return flipped_segms

    def flip_keypoints(self, keypoints, keypoint_flip_map, keypoint_coords, width):
        """Left/right flip keypoint_coords. keypoints and keypoint_flip_map are
        accessible from get_keypoints().
        """
        flipped_kps = keypoint_coords.copy()
        for lkp, rkp in keypoint_flip_map.items():
            lid = keypoints.index(lkp)
            rid = keypoints.index(rkp)
            flipped_kps[:, :, lid] = keypoint_coords[:, :, rid]
            flipped_kps[:, :, rid] = keypoint_coords[:, :, lid]

        # Flip x coordinates
        flipped_kps[:, 0, :] = width - flipped_kps[:, 0, :] - 1
        # Maintain COCO convention that if visibility == 0, then x, y = 0
        inds = np.where(flipped_kps[:, 2, :] == 0)
        flipped_kps[inds[0], 0, inds[1]] = 0
        return flipped_kps

