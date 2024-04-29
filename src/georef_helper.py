import cv2
import numpy as np
import torch

def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

def gray2(img:np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gray3(img:np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray2(img), cv2.COLOR_GRAY2BGR)

def resize_to_max_dim(img:np.ndarray, maxdim:int, return_scale = False) -> (tuple[np.ndarray, float] | np.ndarray):
    h,w,*_ = img.shape
    scale = min(maxdim/h, maxdim/w)
    newd = int(w*scale), int(h*scale)
    rimg = np.array(cv2.resize(img, newd, cv2.INTER_AREA))
    if return_scale:
        return rimg, scale    
    return rimg

def rotate_nocrop(img:np.ndarray, angle:float) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate an image and expand dimensions to avoid cropping.
    
    Parameters:
        img (numpy.ndarray): input image
        angle: rotation angle measured counter-clockwise from the positive x-axis
    
    Return:
        rotated_image (numpy.ndarray): rotated image
        rotation_matrix (numpy.ndarray): 2x3 affine warp matrix
    """
    
    h, w, *_ = img.shape
    R = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    TARGET_WIDTH = int(abs(w*np.cos(angle*np.pi/180)) + abs(h*np.sin(angle*np.pi/180)))
    TARGET_HEIGHT = int(abs(w*np.sin(angle*np.pi/180)) + abs(h*np.cos(angle*np.pi/180)))
    R[0,2] += TARGET_WIDTH//2 - w//2
    R[1,2] += TARGET_HEIGHT//2 - h//2
    return cv2.warpAffine(img, R, (TARGET_WIDTH, TARGET_HEIGHT)), R

def apply_transforms(pt, *ms):
    pt = np.array(pt)
    if len(pt) == 2:
        pt = np.array([*pt,1])
    elif len(pt) == 3:
        pt /= pt[2]
    else:
        raise Exception
    for m in ms:
        pt = m@pt
        pt /= pt[2]
    return pt