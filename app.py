import argparse
import importlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
import torch
# from src.homography import Homography
# from src.georeference import Georeference
from datetime import datetime
from pathlib import Path
import time

'''
    TODO
    - Make the video output in georeference.py optional
    - Traffic analysis
    - Make file creation more robust ('w' instead of 'a')
    - Flesh out app.py (this file) by adding command arg handlers in __name__ section
    - Write a README instruction manual for app.py and src/
'''

torch.set_grad_enabled(False)
# print(os.getcwd())

def generate(vid_path:str, img_path:str, **kwargs):
    """
    Run the georeferencing application on the given video and image.
    
    Parameters:
        vid_in_path (str): Relative or absolute path to input MP4 video.
        img_in_path (str): Relative or absolute path to input GeoTIFF image.
        frame_ini (int): Frame to use as initial frame for output video. Default is 0.
        frame_fin (int): Frame to use as final frame for output video. Default is -1.
        prefix (str): String to use as a folder name for output files.
    """
    path = Path(os.getcwd())
    if not Path(vid_path).is_absolute():
        vid_path = str(path / vid_path)
    if not Path(img_path).is_absolute():
        img_path = str(path / img_path)
    assert os.path.exists(vid_path), f"Video {vid_path} DNE"
    assert os.path.exists(img_path), f"Image {img_path} DNE"
    
    try:
        Homography = getattr(importlib.import_module('src.homography'), 'Homography')
        Georeference = getattr(importlib.import_module('src.georeference'), 'Georeference')
    except:
        Homography = getattr(importlib.import_module('.src.homography', __package__), 'Homography')
        Georeference = getattr(importlib.import_module('.src.georeference', __package__), 'Georeference')
    
    frame_ini = kwargs.get("frame_ini", 0)
    frame_fin = kwargs.get("frame_fin", -1)
    prefix = kwargs.get("prefix")
    sample = kwargs.get("sample")
    if sample < 1:
        sample = 1
    if prefix is None:
        prefix = datetime.now().strftime('%m%d_%H%M%S')
    
    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ini)
    _, IMG_SOURCE = cap.read()
    
    with rasterio.open(img_path) as dat:
        IMG_TARGET = np.moveaxis(np.array(dat.read()),0,-1)[...,:3]
        RWC_M = np.array(dat.transform).reshape((3,3))
    
    hom = Homography(IMG_SOURCE, IMG_TARGET)
    
    if not os.path.exists(f"output/{prefix}"):
        if not os.path.exists(f"output"):
            os.mkdir("output")
        os.mkdir(f"output/{prefix}")
    
    if kwargs.get("intermediates", True):
        plt.imsave(f"output/{prefix}/{prefix}_overlay.png", hom.overlay[:,:,::-1])
        plt.imsave(f"output/{prefix}/{prefix}_matches.png", hom.out)
    
    geo = Georeference(cap, IMG_TARGET, RWC_M, hom.HOM_M, prefix)
    geo.run(frame_ini, frame_fin, sample=sample)

    if kwargs.get("intermediates", True):
        plt.imsave(f"output/{prefix}/{prefix}_speedmap.png", geo.speed_map)
        plt.imsave(f"output/{prefix}/{prefix}_speedoverlay.png", geo.speed_overlay)
        plt.imsave(f"output/{prefix}/{prefix}_volmap.png", geo.volume_map)
        plt.imsave(f"output/{prefix}/{prefix}_sample.png", geo.sample_img[:,:,::-1])

generate("assets/153/153_shake_2.mp4", "assets/153/153_tgt.tif", intermediates=True, sample=1)


if __name__ == "__main__":
    pass
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Georeference from video to satellite imagery, with"
                                                 "potential video and transform text file outputs.")
    parser.add_argument("--vid_path", type=str, required=True,
                        help="Input path to MP4 video.")
    parser.add_argument("--img_path", type=str, required=True,
                        help="Input path to GeoTIFF image.")
    parser.add_argument("--frame_ini", type=int, default=0,
                        help="Frame to use as initial frame for output video.")
    parser.add_argument("--frame_fin", type=int, default=-1,
                        help="Frame to use as final frame of output video. -1 for last frame.")
    parser.add_argument("--prefix", type=str, default=None,
                        help="Optional folder name for saved files")
    parser.add_argument("--sample", type=int, default=1,
                        help="Frame sampling rate for video output")
    parser.add_argument("--intermediates", action='store_true',
                        help="Store intermediate plots of keypoint matching and initial overlay.")
    
    kwargs = parser.parse_args()
    print(vars(kwargs))
    generate(**vars(kwargs))

    end_time = time.time()
    runtime = end_time - start_time

    print("Total run time is: {:.2f} seconds".format(runtime))
