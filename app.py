import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
import torch
from src.homography import Homography
from src.georeference import Georeference
from datetime import datetime

'''
    TODO
    - Make the video output in georeference.py optional
    - Traffic analysis
    - Make file creation more robust ('w' instead of 'a')
    - Flesh out app.py (this file) by adding command arg handlers in __name__ section
    - Write a README instruction manual for app.py and src/
'''

torch.set_grad_enabled(False)

def generate(vid_in_path:str, img_in_path:str, **kwargs):
    """
    Run the georeferencing application on the given video and image.
    
    Parameters:
        vid_in_path (str): Relative or absolute path to input MP4 video.
        img_in_path (str): Relative or absolute path to input GeoTIFF image.
        frame_ini (int): Frame to use as initial frame for output video. Default is 0.
        frame_fin (int): Frame to use as final frame for output video. Default is -1.
        prefix (str): String to use as a folder name for output files.
    """
    assert os.path.exists(vid_in_path), f"Video {vid_in_path} DNE"
    assert os.path.exists(img_in_path), f"Image {img_in_path} DNE"
    
    frame_ini = kwargs.get("frame_ini", 0)
    frame_fin = kwargs.get("frame_fin", -1)
    prefix = kwargs.get("prefix", datetime.now().strftime('%m%d_%H%M%S'))
    
    cap = cv2.VideoCapture(vid_in_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ini)
    _, IMG_SOURCE = cap.read()
    
    with rasterio.open(img_in_path) as dat:
        IMG_TARGET = np.moveaxis(np.array(dat.read()),0,-1)[...,:3]
        RWC_M = np.array(dat.transform).reshape((3,3))
    
    hom = Homography(IMG_SOURCE, IMG_TARGET)
    
    if kwargs.get("intermediates", True):
        if not os.path.exists(f"output/{prefix}"):
            os.mkdir(f"output/{prefix}")
        plt.imsave(f"output/{prefix}/overlay.png", hom.overlay[:,:,::-1])
        plt.imsave(f"output/{prefix}/matches.png", hom.out)
    
    geo = Georeference(cap, IMG_TARGET, RWC_M, hom.HOM_M, prefix)
    geo.run(frame_ini, frame_fin)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Georeference from video to satellite imagery, with"
                                                 "potential video and transform text file outputs.")
    # parser.add_argument("--vid_path", type=str, required=True,
    #                     help="Input path to MP4 video.")
    # parser.add_argument("--img_path", type=str, required=True,
    #                     help="Input path to GeoTIFF image.")
    parser.add_argument("--frame_ini", type=int, default=0,
                        help="Frame to use as initial frame for output video.")
    parser.add_argument("--frame_fin", type=int, default=-1,
                        help="Frame to use as final frame of output video. -1 for last frame.")
    # parser.add_argument("--intermediates", action='store_true',
    #                     help="Store intermediate plots of keypoint matching and initial overlay.")
    
    kwargs = parser.parse_args()
    print(kwargs)

    # TAG = 207
    # generate(vid_in_path = Rf"./assets/{TAG}/{TAG}_vid_destb.mp4",
    #          img_in_path = Rf"./assets/{TAG}/{TAG}_tgt0.tif",
    #          **kwargs)
    
    # generate("assets/UW/pat_destb.mp4", "assets/UW/pat_zoom.tif", **vars(kwargs))
    
    generate("assets/207/207_vid_destb.mp4", "assets/207/207_tgt0.tif", frame_ini=0, frame_fin=600)