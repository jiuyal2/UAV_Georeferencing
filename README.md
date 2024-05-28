### Sponsored by Parkalytics

# UAV Georeferencing
This repository contains a script and relevant objects and methods that tracks vehicles in a video, calculates coordinate transforms, and produces an orthorectified video overlay, given an input MP4 aerial video and GeoTIFF satellite imagery.

### Design Summary
At it's fullest potential, the script first calculates a homography from a frame of the input video to the satellite image, and then runs object tracking and video stabilization in parallel to produce both a visual and text file output of the transformations. Video homography is done with the help of [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork/tree/master), a CNN based keypoint matching library. The input frame and image undergo various rotations and scalings to find an ideal homography. The best transformations, intermediate plots, and resulting homography are saved as state variables. Object detection and tracking is done with the help of [YOLOv8](https://github.com/ultralytics/ultralytics), another CNN that specializes in object detection and tracking. Each video frame is passed into a persistent object tracker so that each car is assigned a unique ID, and the position of each car is mapped to real world coordinates under a specified projection. Video stabilization is done with OpenCV's ORB keypoint detector and FLANN descriptor matching.

### Main Contents

- [`app.py`](app.py) - The main script. It can be run from the command line with the necessary parameters, or imported as a library and run using the generator function.
- [`src/homography.py`](src/homography.py) - Performs homography calculation from a video frame to the satellite image. Internally attempts homographies under various rotations and scalings, and chooses the parameters with highest sum confidence per root match.
- [`src/georeference.py`](src/georeference.py) - Performs homography calculation and object tracking over a range of frames. Saves a list of transforms per frame to a text file, and saves an orthorectified tracking video overlaid on satellite imagery to a video file.

### Usage
First, install any missing dependencies from [`requirements.txt`](requirements.txt). Up-to-date versions of Python and the libraries are recommended.

Sample videos and satellite images are provided in the [`assets`](assets/) folder. You can pass them as relative or absolute file paths parameters to [`app.py`](app.py) in the command line. For example:

`> python app.py --vid_path assets/220/220_vid_destb.mp4 --img_path assets/220/220_tgt.tif`

These parameters are available:

- `--vid_path`: The path to the video file. **Required.**
- `--bar`: bell
- `--fred`: rogers

The program will output a video and txt file to the [`output`](output/) folder.

### Fun xkcd comics

[![In the 60s, Marvin Minsky assigned a couple of undergrads to spend the summer programming a computer to use a camera to identify objects in a scene. He figured they'd have the problem solved by the end of the summer. Half a century later, we're still working on it.](https://imgs.xkcd.com/comics/tasks.png)](https://xkcd.com/1425)

[!['DISPATCHING DRONE TO TARGET COORDINATES.' 'Wait, crap, wrong button. Oh jeez.'](https://imgs.xkcd.com/comics/nro.png)](https://xkcd.com/1358)

[![Cloud computing has a ways to go.](https://imgs.xkcd.com/comics/cloud.png)](https://xkcd.com/1444)