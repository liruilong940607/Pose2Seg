import sys
import numpy as np
import cv2
import json
import math

import lib.transforms as translib

def pose_affinematrix(src_kpt, dst_kpt, dst_area, hard=False):
    ''' `dst_kpt` is the template. 
    Args:
        src_kpt, dst_kpt: (17, 3)
        dst_area: used to uniform returned score.
        hard: 
            - True: for `dst_kpt` is the template. we do not want src_kpt
                to match a template and out of range. So in this case, 
                src_kpt[vis] should convered by dst_kpt[vis]. if not, will 
                return score = 0
            - False: for matching two kpts.
    Returns:
        matrix: (2, 3)
        score: align confidence/similarity, a float between 0 and 1.
    '''
    src_vis = src_kpt[:, 2] > 0
    dst_vis = dst_kpt[:, 2] > 0
    visI = np.logical_and(src_vis, dst_vis)
    visU = np.logical_or(src_vis, dst_vis)
    # - 0 Intersection Points means we know nothing to calc matrix.
    # - 1 Intersection Points means there are infinite matrix.
    # - 2 Intersection Points means there are 2 possible matrix.
    #   But in most case, it will lead to a really bad solution
    if sum(visI) == 0 or sum(visI) == 1 or sum(visI) == 2:
        matrix = np.array([[1, 0, 0], 
                           [0, 1, 0]], dtype=np.float32)
        score = 0.
        return matrix, score
    
    if hard and (False in dst_vis[src_vis]):
        matrix = np.array([[1, 0, 0], 
                           [0, 1, 0]], dtype=np.float32)
        score = 0.
        return matrix, score
      
    src_valid = src_kpt[visI, 0:2]
    dst_valid = dst_kpt[visI, 0:2]
    matrix = solve_affinematrix(src_valid, dst_valid, fullAffine=False)
    matrix = np.vstack((matrix, np.array([0,0,1], dtype=np.float32)))
    
    # calc score
    #sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    #vars_valid = ((sigmas * 2)**2)[visI]
    vars_valid = 1
    diff = translib.warpAffinePoints(src_valid, matrix) - dst_valid
    error = np.sum(diff**2, axis=1) / vars_valid / dst_area / 2
    score = np.mean(np.exp(-error)) * np.sum(visI) / np.sum(visU)
    
    return matrix, score


class PoseAlign():    
    def __init__(self, template_file='templates.json', visualize=True, factor=1.0):
        data = json.load(open(template_file, 'r'))
        
        self.npart = len(data['names'])
        self.vis_threshold = data['vis_threshold']
        self.templates = np.array(data['templates']).reshape(-1, self.npart, 3) # (N, 17, 3)
        inds = np.where(self.templates[:, :, -1] < self.vis_threshold)
        self.templates[inds[0], inds[1], :] = 0
        
        self.templates[:, :, 0:2] = ((self.templates[:, :, 0:2] - 0.5) * factor) + 0.5
        
        # expand templates by left-right flip
        self.templates_flipped = self.flip_keypoints(data['names'], data['flip_map'], self.templates)
        self.templates = np.vstack((self.templates, self.templates_flipped))
        self.templates_category = [1] * len(self.templates) # 0 means full body, 1 means half body
        assert len(self.templates_category) == len(self.templates)
        
        if visualize:
            for center in self.templates:
                center = center.reshape(-1, 3)
                draw_skeleton(center, 200, 200, vis_threshold=0)
                
    def flip_keypoints(self, keypoints, keypoint_flip_map, keypoint_coords):
        """Left/right flip keypoint_coords. keypoints and keypoint_flip_map are
        accessible from get_keypoints().
        """
        flipped_kps = keypoint_coords.copy()
        for lkp, rkp in keypoint_flip_map.items():
            lid = keypoints.index(lkp)
            rid = keypoints.index(rkp)
            flipped_kps[:, lid, :] = keypoint_coords[:, rid, :]
            flipped_kps[:, rid, :] = keypoint_coords[:, lid, :]
        return flipped_kps
    
    def align(self, kpt, srcW, srcH, dstW, dstH, visualize=True, return_history=False):
        # kpt: (17, 3)
        if visualize:
            print ('===> before')
            normed_kpt = norm_kpt_by_box([kpt], [[0, 0, srcW, srcH]], keep_ratio=True)[0]
            draw_skeleton(normed_kpt, h=500, w=500, 
                          vis_threshold=0, is_normed=True)
        
        basic_matrix = translib.get_resize_padding_matrix(srcW, srcH, dstW, dstH, iscenter=True)
        basic_scale = math.sqrt(basic_matrix[0,0] ** 2 + basic_matrix[0,1] ** 2)
        best_dict = {
            'category': -1,
            'template': None,
            'matrix': basic_matrix,
            'score': 0.,
            'history': [],
        }
        for pose, pose_category in zip(self.templates, self.templates_category):
            matrix, score = pose_affinematrix(kpt, pose, dst_area=1.0, hard=True)
            if score > 0:
                # valid `matrix`. default (dstH, dstW) is (1.0, 1.0)
                matrix = translib.get_resize_matrix(1.0, 1.0, dstW, dstH).dot(matrix)
                scale = math.sqrt(matrix[0,0] ** 2 + matrix[0,1] ** 2)
                category = pose_category
            else:
                matrix = basic_matrix
                category = -1
                
            if return_history:
                best_dict['history'].append({
                    'category': category,
                    'template': pose,
                    'matrix': matrix,
                    'score': score
                })
            
            if score > best_dict['score']:
                best_dict['category'] = category
            
            if score > best_dict['score'] and scale > basic_scale:
                best_dict['template'] = pose
                best_dict['matrix'] = matrix
                best_dict['score'] = score
            
            
            if visualize and score > 0:
                print (score)
                draw_skeleton(translib.warpAffineKpts([kpt], matrix)[0], h=dstH, w=dstW, 
                          vis_threshold=0, is_normed=False)
                print ('===> use template')
                draw_skeleton(pose, h=500, w=500, 
                              vis_threshold=0, is_normed=True)
            
        if visualize:
            print ('===> after')
            draw_skeleton(translib.warpAffineKpts([kpt], best_dict['matrix'])[0], h=dstH, w=dstW, 
                          vis_threshold=0, is_normed=False)
            if best_dict['template'] is not None:
                print ('===> use template')
                draw_skeleton(best_dict['template'], h=500, w=500, 
                              vis_threshold=0, is_normed=True)

        return best_dict

def solve_affinematrix(src, dst, fullAffine=False):
    '''
    Document: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html?highlight=solve#cv2.solve
    C++ Version: aff_trans.cpp in opencv
    src: numpy array (N, 2)
    dst: numpy array (N, 2)
    fullAffine = False means affine align without shear.
    '''
    src = src.reshape(-1, 1, 2)
    dst = dst.reshape(-1, 1, 2)
    
    out = np.zeros((2,3), np.float32)
    siz = 2*src.shape[0]

    if fullAffine:
        matM = np.zeros((siz,6), np.float32)
        matP = np.zeros((siz,1), np.float32)
        contPt=0
        for ii in range(0, siz):
            therow = np.zeros((1,6), np.float32)
            if ii%2==0:
                therow[0,0] = src[contPt, 0, 0] # x
                therow[0,1] = src[contPt, 0, 1] # y
                therow[0,2] = 1
                matM[ii,:] = therow[0,:].copy()
                matP[ii,0] = dst[contPt, 0, 0] # x
            else:
                therow[0,3] = src[contPt, 0, 0] # x
                therow[0,4] = src[contPt, 0, 1] # y
                therow[0,5] = 1
                matM[ii,:] = therow[0,:].copy()
                matP[ii,0] = dst[contPt, 0, 1] # y
                contPt += 1

        sol = cv2.solve(matM, matP, flags = cv2.DECOMP_SVD)
        sol = sol[1]
        out = sol.reshape(2, -1)
    else:
        matM = np.zeros((siz,4), np.float32)
        matP = np.zeros((siz,1), np.float32)
        contPt=0
        for ii in range(0, siz):
            therow = np.zeros((1,4), np.float32)
            if ii%2==0:
                therow[0,0] = src[contPt, 0, 0] # x
                therow[0,1] = src[contPt, 0, 1] # y
                therow[0,2] = 1
                matM[ii,:] = therow[0,:].copy()
                matP[ii,0] = dst[contPt, 0, 0] # x
            else:
                therow[0,0] = src[contPt, 0, 1] # y ## Notice, c++ version is - here
                therow[0,1] = -src[contPt, 0, 0] # x
                therow[0,3] = 1
                matM[ii,:] = therow[0,:].copy()
                matP[ii,0] = dst[contPt, 0, 1] # y
                contPt += 1
        sol = cv2.solve(matM, matP, flags = cv2.DECOMP_SVD)
        sol = sol[1]
        out[0,0]=sol[0,0]
        out[0,1]=sol[1,0]
        out[0,2]=sol[2,0]
        out[1,0]=-sol[1,0]
        out[1,1]=sol[0,0]
        out[1,2]=sol[3,0]

    # result
    return out

def norm_kpt_by_box(kpts, boxes, keep_ratio=True):
    normed_kpts = np.array(kpts).copy()
    normed_kpts = np.float32(normed_kpts)
    
    for i, (kpt, box) in enumerate(zip(kpts, boxes)):
        H = translib.get_cropalign_matrix(box, 1.0, 1.0, keep_ratio)
        normed_kpts[i, :, 0:2] = translib.warpAffinePoints(kpt[:, 0:2], H)
        
    inds = np.where(normed_kpts[:, :, 2] == 0)
    normed_kpts[inds[0], inds[1], :] = 0
    return normed_kpts

def draw_skeleton(normed_kpts, h=200, w=200, vis_threshold=0, is_normed=True, returnimg=False):
    import matplotlib.pyplot as plt
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