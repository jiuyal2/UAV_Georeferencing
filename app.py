import cv2
import numpy as np
import os
import rasterio
import torch
from src.homography import Homography
from src.georeference import Georeference

torch.set_grad_enabled(False)

def generate(vid_in_path, img_in_path, **kwargs):
    assert os.path.exists(vid_in_path), print("Video DNE")
    assert os.path.exists(img_in_path), print("Image DNE")
    
    cap = cv2.VideoCapture(vid_in_path)
    ini = kwargs["ini"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, ini)
    _, IMG_SOURCE = cap.read()
    
    with rasterio.open(img_in_path) as dat:
        IMG_TARGET = np.moveaxis(np.array(dat.read()),0,-1)[...,:3]
        RWC_M = np.array(dat.transform).reshape((3,3))
    
    hom = Homography(IMG_SOURCE, IMG_TARGET)
    geo = Georeference(cap, IMG_TARGET, RWC_M, hom.HOM_M, "2")
    geo.run(ini,500)
    
    
    '''
    cap = get the video
    get first frame user wants
    tif = get the tif file
    get tif image
    homographize it with frame
    run the georeference class
        * app.py NEVER sees RWC_M directly, it's just a wrapper for
          all the class gruntwork
          * what app.py *does* do is check files exist, make sure parameters are
            correct, uses cmdline to ask for more info if needed, 
        - which runs the video writer class if we want visual output
        - but will output .... what? ... regardless of video or not
    '''

if __name__ == "__main__":
    # parse args
    kwargs = dict(ini=0, fin=20)

    TAG = 220
    generate(Rf"./assets/{TAG}/{TAG}_vid_destb.mp4", Rf"./assets/{TAG}/{TAG}_tgt.tif",
             **kwargs)

    '''
    TODO
    - Get a second transforms.txt file specific to the video
    - Condense all the lines
    - Redo file structure
    - 
    '''