import json
import numpy as np
from tqdm import tqdm
from scipy.cluster import vq
import cv2
import matplotlib.pyplot as plt

from datasets.CocoDatasetInfo import CocoDatasetInfo
from lib.transforms import get_cropalign_matrix, warpAffinePoints

def draw_skeleton(normed_kpts, h=200, w=200, vis_threshold=0, is_normed=True, returnimg=False):
    origin_connections = [[16,14],[14,12],[17,15],[15,13],[12,13],
                      [6,12],[7,13],[6,7],[6,8],[7,9],[8,10],
                      [9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    img = np.zeros((int(h), int(w)), dtype=np.float32)
    kptsv = normed_kpts.copy()
    if is_normed:
        kptsv[:, 0] *= w
        kptsv[:, 1] *= h
    kptsv = np.int32(kptsv)
    
    for kptv in kptsv:
        if kptv[-1] > vis_threshold:
            cv2.circle(img, (kptv[0], kptv[1]), 4, (255, 0, 0), -1)
    idx = 15
    cv2.circle(img, (kptsv[idx][0], kptsv[idx][1]), 10, (0, 0, 255), -1)
    for conn in origin_connections:
        if kptsv[conn[0] - 1][-1] > vis_threshold and kptsv[conn[1] - 1][-1] > vis_threshold:
            p1, p2 = kptsv[conn[0] - 1], kptsv[conn[1] - 1]
            cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 0), 2)
            
    if returnimg:
        return img
    else:
        plt.imshow(img)
        plt.show()
        
def norm_kpt_by_box(kpts, boxes, keep_ratio=True):
    normed_kpts = np.array(kpts).copy()
    normed_kpts = np.float32(normed_kpts)
    
    for i, (kpt, box) in enumerate(zip(kpts, boxes)):
        H = get_cropalign_matrix(box, 1.0, 1.0, keep_ratio)
        normed_kpts[i, :, 0:2] = warpAffinePoints(kpt[:, 0:2], H)
        
    inds = np.where(normed_kpts[:, :, 2] == 0)
    normed_kpts[inds[0], inds[1], :] = 0
    return normed_kpts

def cluster_zixi(kpts, cat_num):
    # kpts: center-normalized (N, 17, 3)    
    datas = np.array(kpts)
    inds = np.where(datas[:, :, 2] == 0)
    datas[inds[0], inds[1], 0:2] = 0.5
    
    datas = datas.reshape(len(datas), -1)
    res = vq.kmeans2(datas, cat_num, minit='points', iter=100)
    return res

def cluster(dataset = 'coco', cat_num = 3, vis_threshold = 0.4, 
            minpoints = 8,  save_file = './modeling/templates2.json', visualize=False):
    # We try `cat_num` from 1 to 6 multiple times. we want to see
    # what the cluster centers look like when vary the numbers of
    # group. While the kmean method, which is heavily relay on the 
    # initial status, gives nearly the same cluster centers when
    # `cat_num` = 3 each time. So we assume the coco dataset accurately
    # have 3 clusters.(a TODO is to visualize this dataset.) And 
    # the visualization of the cluster centers seems to reasonable: 
    # (1) a full body. (2) a full body without head (3) an upper body.
    # Note that (2) seems representing the backward of a person.
    
    if dataset == 'coco':
        datainfos = CocoDatasetInfo('./data/coco2017/train2017',
                                   './data/coco2017/annotations/person_keypoints_train2017_pose2seg.json',
                                   loadimg=False)
        
        connections = [[16,14],[14,12],[17,15],[15,13],[12,13],
                       [6,12],[7,13],[6,7],[6,8],[7,9],[8,10],
                       [9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
        
        names = ["nose",
                 "left_eye","right_eye",
                 "left_ear","right_ear",
                 "left_shoulder","right_shoulder",
                 "left_elbow","right_elbow",
                 "left_wrist","right_wrist",
                 "left_hip","right_hip",
                 "left_knee","right_knee",
                 "left_ankle","right_ankle"]
        
        flip_map = {'left_eye': 'right_eye',
                    'left_ear': 'right_ear',
                    'left_shoulder': 'right_shoulder',
                    'left_elbow': 'right_elbow',
                    'left_wrist': 'right_wrist',
                    'left_hip': 'right_hip',
                    'left_knee': 'right_knee',
                    'left_ankle': 'right_ankle'}
        
        def flip_keypoints(keypoints, keypoint_flip_map, keypoint_coords, width):
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
            flipped_kps[:, 0, :] = width - flipped_kps[:, 0, :]
            # Maintain COCO convention that if visibility == 0, then x, y = 0
            inds = np.where(flipped_kps[:, 2, :] == 0)
            flipped_kps[inds[0], 0, inds[1]] = 0
            return flipped_kps

        all_kpts = []
        for idx in tqdm(range(len(datainfos))):
            rawdata = datainfos[idx]
            gt_boxes = rawdata['boxes']
            gt_kpts = rawdata['gt_keypoints'].transpose(0, 2, 1) # (N, 17, 3)
            gt_ignores = rawdata['is_crowd']
            normed_kpts = norm_kpt_by_box(gt_kpts, gt_boxes)
            normed_kpts_flipped = flip_keypoints(names, flip_map,
                                                normed_kpts.transpose(0, 2, 1), 1.0).transpose(0, 2, 1)
            normed_kpts = np.vstack((normed_kpts, normed_kpts_flipped))
            for kpt in normed_kpts:
                if np.sum(kpt)==0:
                    continue
                elif np.sum(kpt[:, 2]>0)<minpoints:
                    continue
                else:
                    all_kpts.append(kpt)
        all_kpts = np.array(all_kpts)
        print ('data to be clustered:', all_kpts.shape) 
        
        res = cluster_zixi(all_kpts, cat_num)
        
        save_dict = {}
        save_dict['connections'] = connections
        save_dict['names'] = names
        save_dict['flip_map'] = flip_map
        save_dict['vis_threshold'] = vis_threshold
        save_dict['minpoints'] = minpoints
        save_dict['templates'] = [item.tolist() for item in res[0]]
        if save_file is not None:
            with open(save_file, 'w') as result_file:
                json.dump(save_dict, result_file)
        
        if visualize:
            for center in res[0]:
                center = center.reshape(-1, 3)
                draw_skeleton(center, 200, 200, vis_threshold)
        
        print ('cluster() done.')
        return res
    
    else:
        raise NotImplementedError
