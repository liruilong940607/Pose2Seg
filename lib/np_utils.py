import numpy as np
import cv2

def masks2bboxes(masks):
    '''
    masks: (N, H, W) or [N](H, W)
    '''
    bboxes = []
    for mask in masks:
        if np.max(mask)<=0.5:
            continue
        idxs = np.where(mask>0.5)
        ymax = np.max(idxs[0])
        ymin = np.min(idxs[0])
        xmax = np.max(idxs[1])
        xmin = np.min(idxs[1])
        bboxes.append([xmin, ymin, xmax, ymax])
    bboxes = np.array(bboxes, dtype=np.float32)
    return bboxes


def resize_keep_ratio(img, size, mode=0, interpolation=cv2.INTER_LINEAR):
    r"""Resize the input Image to the given size.
    Args:
        img (Array Image): Image to be resized.
        size (int): Desired output size.
        mode (int, optional): Desired mode. 
            if mode=='max', max(w, h) ->  size
            if mode=='min', min(w, h) ->  size
            if mode=='mean', mean(w, h) ->  size
            Default is 0
    Returns:
        Array Image: Resized image.
    """
    assert mode in ['max', 'min', 'mean'], \
        'Resize_keep_ratio mode should be either max, min, or mean'
        
    srcH, srcW = img.shape[0:2]
    if (srcW < srcH and mode == 'max') or (srcW > srcH and mode == 'min'):
        dstH = size
        dstW = int(float(size) * srcW / srcH)
    elif (srcW > srcH and mode == 'max') or (srcW < srcH and mode == 'min'):
        dstH = size
        dstW = int(float(size) * srcW / srcH)
    else: # mode == 'mean'
        scale = np.mean((srcH, srcW)) / size
        dstH, dstW = [srcH*scale, srcW*scale]
    
    return cv2.resize(img, (dstW, dstH), interpolation)


def pad(img, padding, value=0, borderType=cv2.BORDER_CONSTANT):
    '''
    Based on `cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)`
    '''
    return cv2.copyMakeBorder(img, padding[0], padding[1], padding[2], padding[3], borderType, value=value)


def pad_to(img, h, w, iscenter=False, value=0, borderType=cv2.BORDER_CONSTANT):
    deltay = int(h - img.shape[0])
    deltax = int(w - img.shape[1])
    assert deltax>=0 and deltay>=0
    if iscenter:
        # top, bottom, left, right
        padding = (int(deltay/2), deltay-int(deltay/2), 
                   int(deltax/2), deltay-int(deltax/2))
    else:
        padding = (0, deltay, 0, deltax)
    img = cv2.copyMakeBorder(img, padding[0], padding[1], padding[2], padding[3], borderType, value=value)
    return img
    

def resize_padding(img, dstH, dstW, minsize=0, maxsize=0, padvalue=0, iscenter=False, interpolation=cv2.INTER_LINEAR):
    height, width = img.shape[0:2]
    dtype = img.dtype
    img = np.float32(img)
    if minsize>0 and maxsize>0:
        # minsize <= dstH, dstW <= maxsize
        im_minsize = min(height, width)
        im_maxsize = max(height, width)
        scale = min(float(minsize)/im_minsize, float(maxsize)/im_maxsize)
        img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=interpolation)
    else:
        scale = min(float(dstH)/height, float(dstW)/width)
        img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=interpolation)
        assert img.shape[0]==round(scale*height)
        assert img.shape[1]==round(scale*width)
        img = pad_to(img, dstH, dstW, iscenter, value=padvalue)
    img = img.astype(dtype)
    return img, scale

def draw_boxes(img, boxes, color=(255, 255, 255), thickness=3):
    # (x1, y1, x2, y2)
    canvas = img.copy()
    for box in boxes:
        box = np.array(box, dtype=np.int32)
        cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, thickness)
    return canvas


    