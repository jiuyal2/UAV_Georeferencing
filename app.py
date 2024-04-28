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

def generate(vid_in_path, img_in_path, **kwargs):
    '''
    * app.py doesn't use RWC_M directly, this method is just a wrapper for
        all the class gruntwork
        * what app.py *does* do is check files exist, make sure parameters are
        correct, uses cmdline to ask for more info if needed, 
    - but will output .... what? ... regardless of video or not
    '''
    assert os.path.exists(vid_in_path), print("Video DNE")
    assert os.path.exists(img_in_path), print("Image DNE")
    
    cap = cv2.VideoCapture(vid_in_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, kwargs["frame_ini"])
    _, IMG_SOURCE = cap.read()
    
    with rasterio.open(img_in_path) as dat:
        IMG_TARGET = np.moveaxis(np.array(dat.read()),0,-1)[...,:3]
        RWC_M = np.array(dat.transform).reshape((3,3))
    
    hom = Homography(IMG_SOURCE, IMG_TARGET)
    
    plt.imsave(f"output/{datetime.now().strftime('%m%d_%H%M%S')}_overlay.png", hom.overlay[:,:,::-1])
    plt.imsave(f"output/{datetime.now().strftime('%m%d_%H%M%S')}_matches.png", hom.out)
    
    geo = Georeference(cap, IMG_TARGET, RWC_M, hom.HOM_M, "207")
    geo.run(kwargs["frame_ini"], kwargs["frame_fin"])


if __name__ == "__main__":
    # parse args
    kwargs = dict(frame_ini=0, frame_fin=360)

    TAG = 207
    generate(vid_in_path = Rf"./assets/{TAG}/{TAG}_vid_destb.mp4",
             img_in_path = Rf"./assets/{TAG}/{TAG}_tgt0.tif",
             **kwargs)