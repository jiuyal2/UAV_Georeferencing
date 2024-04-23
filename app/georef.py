import csv
import cv2
import numpy as np
import os
import rasterio
import torch
import threading
import queue
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from video_writer import Homographize
from lib.superglue.matching import Matching
from lib.superglue.futils import make_matching_plot_fast

torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}

matching = Matching(config).eval().to(device)
keys = ['keypoints', 'scores', 'descriptors']

TAR_SM_DIM = 480
SRC_SM_DIM = TAR_SM_DIM
TAR_LG_DIM = 720
SRC_LG_DIM = TAR_LG_DIM

### HELPER FUNCTIONS

def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

gray2 = lambda img : cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray3 = lambda img : cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

def resize_to_max_dim(img, maxdim):
    h,w,*_ = img.shape
    scale = min(maxdim/h, maxdim/w)
    newd = int(w*scale), int(h*scale)
    return cv2.resize(img, newd, cv2.INTER_AREA), scale

def rotate_nocrop(img, angle):
    h, w, *_ = img.shape
    R = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    pi = 3.141592
    TARGET_WIDTH = int(abs(w*np.cos(angle*pi/180)) + abs(h*np.sin(angle*pi/180)))
    TARGET_HEIGHT = int(abs(w*np.sin(angle*pi/180)) + abs(h*np.cos(angle*pi/180)))
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

### END HELPER FUNCTIONS

def generate(vid_in_path:str,
             img_target_path:str,
             file_out_path:str = "transforms.txt",
             vid_out_path:str = "georef.mp4"):
    """
    Generate transform-per-frame and warped video output.
    
    Parameters:
        vid_in_path (str): The path to the *.mp4 video.
        img_target_path (str): The path to the *.tif/*.tiff file. Must include projection and transform.
        file_out_path (str): The output filename for the list of transforms per frame.
        vid_out_path (str): The name of the output video. `None` if no output.
    """
    
    # Steps:
    # 1 - strip the first frame of the video
    
    cap = cv2.VideoCapture(vid_in_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, IMG_SOURCE = cap.read()
    
    # 2 - import tiff file and data, do resizes
    
    with rasterio.open(img_target_path) as dat:
        IMG_TARGET = np.moveaxis(np.array(dat.read()),0,-1)
        RWC_M = np.array(dat.transform).reshape((3,3))
        
    img_source_sm,_ = resize_to_max_dim(IMG_SOURCE, SRC_SM_DIM)
    img_target_sm,_ = resize_to_max_dim(IMG_TARGET, TAR_SM_DIM)
    img_source_sm_gr = gray2(img_source_sm)
    img_target_sm_gr = gray2(img_target_sm)

    img_source_lg,s1 = resize_to_max_dim(IMG_SOURCE, SRC_LG_DIM)
    img_target_lg,s2 = resize_to_max_dim(IMG_TARGET, TAR_LG_DIM)
    img_source_lg_gr = gray2(img_source_lg)
    img_target_lg_gr = gray2(img_target_lg)
    
    # 3 - run homography calculations, store homographies
    
    ### ANGLE FINDING

    frame_tensor = frame2tensor(img_source_sm_gr, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor

    print(".", end="")

    best = [0,0,0]

    for ang in range(0, 360, 60):
        img_target_sm_gr_rot, rot_M = rotate_nocrop(img_target_sm_gr, ang)
        
        frame_tensor = frame2tensor(img_target_sm_gr_rot, device)
        pred = matching({**last_data, 'image1': frame_tensor})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()

        print(".", end="")

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        color = cm.jet(confidence[valid])

        print(".", len(mkpts0), np.sum(confidence[valid]), np.sum(confidence[valid])/len(mkpts0))
        
        if np.sum(confidence[valid])/(1+len(mkpts0))**0.5 > best[1]:
            best[0] = ang
            best[1] = np.sum(confidence[valid])/(1+len(mkpts0))**0.5
            
    print(best)
    
    ### USING BEST ROTATION

    frame_tensor = frame2tensor(img_source_lg_gr, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor

    print(".", end="")

    img_target_lg_gr_rot, rot_M = rotate_nocrop(img_target_lg_gr, best[0])
    frame_tensor = frame2tensor(img_target_lg_gr_rot, device)
    pred = matching({**last_data, 'image1': frame_tensor})
    kpts0 = last_data['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()

    print(".", end="")

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    color = cm.jet(confidence[valid])
    
    out = make_matching_plot_fast(
        img_source_lg_gr, img_target_lg_gr_rot, kpts0, kpts1, mkpts0, mkpts1, color, text=[],
        path=None, show_keypoints=True, small_text=[])
    plt.imsave("matches.png", out)

    print(".", len(mkpts0), end="\n")
    
    ### CALCULATED HOMOGRAPHY

    if len(mkpts0) < 10:
        raise Exception
    hom_M, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
    h,w,*_ = img_target_lg_gr_rot.shape
    img_source_warp = cv2.warpPerspective(img_source_lg_gr, hom_M, (w, h))
    h,w,*_ = img_target_lg_gr.shape
    img_source_warp = cv2.warpAffine(img_source_warp, rot_M, (w,h), flags=cv2.WARP_INVERSE_MAP)
    mask = (img_source_warp > 0)
    img_overlay_gray = img_target_lg_gr.copy()
    img_overlay_gray[mask] = img_source_warp[mask]

    # 4 - Loop through each frame of the video and calculate stb_M, save to txt.
    
    if vid_out_path is not None:
        pass
    hh = Homographize(img_target_lg, vid_in_path, vid_out_path, file_out_path,
                [s1, s2], hom_M, rot_M, RWC_M, frame_fin=400)
    hh.run()
    # 4.5 - If output video, then save frames to buffer. Else, discard.