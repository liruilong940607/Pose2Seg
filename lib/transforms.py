import math
import numpy as np
import random

# ** core **
def get_affine_matrix(center, angle, translate, scale, shear=0):
    # Helper method to compute affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a)*sx    -sin(a + shear)*sy     0]
    #                              [ sin(a)*sx    cos(a + shear)*sy     0]
    #                              [     0                  0          1]

    angle = math.radians(angle)
    shear = math.radians(shear)

    T = np.array([[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]]).astype(np.float32)
    C = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]]).astype(np.float32)
    RSS = np.array([[ math.cos(angle)*scale[0], -math.sin(angle + shear)*scale[1], 0],
                    [ math.sin(angle)*scale[0],  math.cos(angle + shear)*scale[1], 0],
                    [ 0, 0, 1]]).astype(np.float32)
    C_inv = np.linalg.inv(np.mat(C))
    M = T.dot(C).dot(RSS).dot(C_inv)
    return M

# ** tools **
def get_aug_matrix(srcW, srcH, dstW, dstH, angle_range=(-45, 45), scale_range=(0.5, 1.5), trans_range=(-0.3, 0.3)):
    center = (srcW/2.0, srcH/2.0)  
    init_scale = min(float(dstW)/srcW, float(dstH)/srcH)

    angle = random.random()*(angle_range[1]-angle_range[0])+angle_range[0]
    sx = sy = random.random()*(scale_range[1]-scale_range[0])+scale_range[0]
    scale = (sx*init_scale, sy*init_scale)
    tx = random.random()*(trans_range[1]-trans_range[0])+trans_range[0]
    ty = random.random()*(trans_range[1]-trans_range[0])+trans_range[0]
    translate = (tx*dstW + (dstW-srcW)/2, ty*dstH + (dstH-srcH)/2)
    
    H = get_affine_matrix(center, angle, translate, scale, shear=0)
    params = {'center':center, 'angle':angle, 'translate':translate, 'scale':scale, 'shear':0}
    return H, params


def warpAffinePoints(pts, H):
    # pts: (N, (x,y))
    pts = np.array(pts, dtype=np.float32)
    assert H.shape in [(3,3), (2,3)], 'H.shape must be (2,3) or (3,3): {}'.format(H.shape)
    ext = np.ones((len(pts), 1), dtype=pts.dtype)
    return np.array(np.hstack((pts, ext)).dot(H[0:2, :].transpose(1, 0)), dtype=np.float32)

def warpAffineKpts(kpts, H):
    # kpts: (N, 17, 3)
    warped_kpts = np.array(kpts)
    warped_kpts[:, :, 0:2] = warpAffinePoints(warped_kpts[:, :, 0:2].reshape(-1, 2), H).reshape(-1, 17, 2)
    inds = np.where(warped_kpts[:, :, 2] == 0)
    warped_kpts[inds[0], inds[1], :] = 0
    return warped_kpts

def warpAffineBoxes(boxes, H, outer=False):
    # pts: (N, (x1,y1,x2,y2))
    assert H.shape in [(3,3), (2,3)], 'H.shape must be (2,3) or (3,3): {}'.format(H.shape)
    boxes = np.array(boxes, dtype=np.float32)
    if outer==False:
        assert H[0,1] == H[1,0] == 0, 'warpAffineBoxes(outer=False) do not support rotation: {}'.format(H)
        pts1 = warpAffinePoints(boxes[:, 0:2], H)
        pts2 = warpAffinePoints(boxes[:, 2:4], H)
        return np.hstack((pts1,pts2))
    else:
        pts1 = warpAffinePoints(boxes[:, 0:2], H)
        pts2 = warpAffinePoints(boxes[:, 2:4], H)
        pts3 = warpAffinePoints(boxes[:, [0,3]], H)
        pts4 = warpAffinePoints(boxes[:, [2,1]], H)
        xs = np.hstack((pts1[:, 0:1], pts2[:, 0:1], pts3[:, 0:1], pts4[:, 0:1]))
        ys = np.hstack((pts1[:, 1:2], pts2[:, 1:2], pts3[:, 1:2], pts4[:, 1:2]))
        xmin, xmax = [np.min(xs, axis = 1, keepdims=True), np.max(xs, axis = 1, keepdims=True)]
        ymin, ymax = [np.min(ys, axis = 1, keepdims=True), np.max(ys, axis = 1, keepdims=True)]
        return np.hstack((xmin, ymin, xmax, ymax))

def get_cropalign_matrix(box, dstW, dstH, keep_ratio=False):
    # box: [x1, y1, x2, y2]
    cropM = get_crop_matrix(box)
    srcW = box[2] - box[0]
    srcH = box[3] - box[1]
    if keep_ratio:
        alignM = get_resize_padding_matrix(srcW, srcH, dstW, dstH, iscenter=True)
    else:
        alignM = get_resize_matrix(srcW, srcH, dstW, dstH)
    return alignM.dot(cropM)
    
def get_crop_matrix(box):
    # box: [x1, y1, x2, y2]
    # This function simply translate the coordinate.
    return np.array([[1, 0, -box[0]],
                     [0, 1, -box[1]],
                     [0, 0, 1]], dtype=np.float32)

def get_resize_padding_matrix(srcW, srcH, dstW, dstH, iscenter=False):
    # this function keep ratio
    scalex = scaley = min(float(dstW)/srcW, float(dstH)/srcH)
    if iscenter:
        translate = ((dstW - srcW * scalex)/2.0, (dstH - srcH * scaley)/2.0)
    else:
        translate = (0, 0)
    return get_affine_matrix(center=(0, 0), angle=0, translate=translate, scale=(scalex, scaley))

def get_resize_matrix(srcW, srcH, dstW, dstH):
    # this function do not keep ratio
    scalex, scaley = (float(dstW)/srcW, float(dstH)/srcH)
    return get_affine_matrix(center=(0, 0), angle=0, translate=(0, 0), scale=(scalex, scaley))

def xfilp_matrix(srcW):
    return np.array([[-1, 0, srcW],
                     [ 0, 1, 0],
                     [ 0, 0, 1]], dtype=np.float32)

def stride_matrix(factor):
    return np.array([[factor,  0, 0],
                     [ 0, factor, 0],
                     [ 0,      0, 1]], dtype=np.float32)











