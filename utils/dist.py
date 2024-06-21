import numpy as np

def iou(bbox, candidate):
    '''Computing intersection over union
    codes from https://github.com/nwojke/deep_sort/blob/master/deep_sort/iou_matching.py
    
    Parameters
    ----------
    bbox : ndarray
        A bbox in format `(x,y,w,h)`
    candidate : ndarray
        A candidate bboxes `(n,4)
    '''
    
    bbox_tl, bbox_br = bbox[:2], bbox[:2]+bbox[2:]
    candidate_tl, candidate_br = candidate[:,:2], candidate[:,:2]+candidate[:,2:]
    
    tl = np.c_[np.maximum(bbox_tl[0], candidate_tl[:, 0])[:, None], 
          np.maximum(bbox_tl[1], candidate_tl[:, 1])[:, None]]
    
    br = np.c_[np.minimum(bbox_br[0], candidate_br[:, 0])[:, None], 
          np.minimum(bbox_br[1], candidate_br[:, 1])[:, None]]
    
    wh = np.maximum(0, br-tl)
    
    area_intersection = np.prod(wh, axis=1)
    area_bbox = bbox[2:].prod(axis=1)
    area_candidates = candidate[:, 2:].prod(axis=1)
    
    return area_intersection / (area_bbox+area_candidates-area_intersection)
