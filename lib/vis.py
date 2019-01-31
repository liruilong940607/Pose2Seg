import numpy as np
import cv2

def _uniform(arr, _min=None, _max=None):
    height, width = arr.shape[0:2]
    _min = np.min(arr) if _min is None else _min
    _max = np.max(arr) if _max is None else _max
    if _min == _max:
        return np.zeros((height, width, 3), dtype=np.uint8)
    vis = np.clip(arr, _min, _max)
    vis = (vis - _min) / (_max - _min) 
    vis = np.uint8(vis * 255.0)
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    return vis

def _hstack(arr_list, height=400):
    arrs = []
    for arr in arr_list:
        width = round(height / arr.shape[0] * arr.shape[1])
        canvas = cv2.resize(arr, (width, height))
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], canvas.shape[0]), color=(255, 160, 122), thickness=2)
        arrs.append(canvas)
    return np.hstack(tuple(arrs))

def _vstack(arr_list):
    max_width = max([arr.shape[1] for arr in arr_list])
    arrs = []
    for arr in arr_list:
        pad_right = max_width - arr.shape[1]
        # top, bottom, left, right
        canvas = cv2.copyMakeBorder(arr, 0, 0, 0, pad_right, cv2.BORDER_CONSTANT, value=(255, 160, 122))
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], canvas.shape[0]), color=(255, 160, 122), thickness=2)
        arrs.append(canvas)
    return np.vstack(tuple(arrs))