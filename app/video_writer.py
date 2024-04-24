import threading
import cv2
import os
import queue
from tqdm import tqdm
import numpy as np
from detection import DetectionModel


### HOMOGRAPHY AND VIDEO WRITING CLASS

class VideoWriterThread(threading.Thread):
    def __init__(self, queue, vid_out_path, fps, width, height):
        super(VideoWriterThread, self).__init__()
        self.queue = queue
        self.vid_out_path = vid_out_path
        self.fps = fps
        self.width = width
        self.height = height

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_stack = cv2.VideoWriter(self.vid_out_path, fourcc, self.fps, (self.width, self.height))

        while True:
            frame = self.queue.get() # blocking, so will wait until queue not empty, return immediately on new entry
            if frame is None:
                break
            out_stack.write(frame)

        out_stack.release()


class Homographize():
    def __init__(self, im_tar, vid_in_path, vid_out_fn = None, transform_fn = None,
                 scales = None, hom_M = None, rot_M = None, ref_M = None,
                 frame_ini = 0, frame_fin = -1):
        """
        Rotates a video onto a satellite image given a set of transforms.
        
        Parameters:
            im_tar (numpy.ndarray): The target satellite image. If rescaled from original,
                provide scale factor in `scales` parameter.
            vid_in_path (str): Path to input video.
            vid_out_fn (str): Output video filename path. Include .mp4 suffix.
            transform_fn (str): Output transform matrices filename path. Include .txt suffix.
            scales (list[float]): Scale factors for video frames and satellite image.
            hom_M (numpy.ndarray): The 3x3 homography matrix from video to satellite for scaled images.
            rot_M (numpy.ndarray): The 2x3 rotation matrix applied to the satellite image before keypoint detection.
            ref_M (numpy.ndarray): The 3x3 transform matrix from satellite pixels to real world coordinates.
            frame_ini (int): The frame to use as the first frame of the output video.
            frame_fin (int): The frame to use as the last frame of the output video. -1 for all frames.
            
        Output:
            Video file of input video rotated onto satellite image.
            Text file of transform matrices from unscaled video to scaled satellite image.
            Text file of transform matrices from unscaled video to real world coordinates.
        """

        self.vid_in_path = vid_in_path
        cwd = os.getcwd()
        if vid_out_fn == None:
            file_name = os.path.splitext(vid_in_path)[0]
            self.vid_out_path = os.path.join(cwd, file_name+'_stb.mp4')
        else:
            self.vid_out_path = os.path.join(cwd, vid_out_fn)
        self.transform_fn = transform_fn

        self.cap = cv2.VideoCapture(self.vid_in_path)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.im_tar = im_tar
        self.h, self.w, _ = im_tar.shape
        
        self.scales = scales
        self.hom_M = hom_M # 3x3
        self.rot_M = rot_M # 2x3
        self.ref_M = ref_M # 3x3

        self.frame_ini = frame_ini
        if frame_fin == -1:
            self.frame_fin = self.n_frames-1
        else:
            self.frame_fin = frame_fin

        self.queue = queue.Queue()
        self.video_writer_thread = VideoWriterThread(self.queue, self.vid_out_path, self.fps, self.w, self.h)

    def run(self):
        self.video_writer_thread.start()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_ini)
        success, prev_img = self.cap.read()
        assert success == True, f"Problems reading file: {self.vid_in_path}"
        
        h_og, w_og, *_ = prev_img.shape
        h_nw, w_nw = int(h_og*self.scales[0]), int(w_og*self.scales[0])
        # these new dimensions carry over for all resizes of "prev_img"
        prev_img = cv2.resize(prev_img, (w_nw, h_nw), cv2.INTER_AREA)
        
        orb = cv2.ORB_create()
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        prev_kp, prev_des = orb.detectAndCompute(prev_img, mask=None)

        ##############################################################################################
        # detection interface
        det_model = DetectionModel(self.vid_in_path)
        ##############################################################################################

        for i in tqdm(range(self.frame_ini,self.frame_fin)):
            # this entire thing may need a try except so the video writer can be released
            success, curr_img = self.cap.read()
            if not success:
                break
            
            ##########################################################################################
            # detection interface
            det_model.next_frame()
            coordinates = det_model.get_coordinate()
            ids = det_model.get_id()
            cls = det_model.get_class()
            ##########################################################################################

            curr_img = cv2.resize(curr_img, (w_nw, h_nw), cv2.INTER_AREA)
            
            curr_kp, curr_des = orb.detectAndCompute(curr_img, None)
            matches = flann.knnMatch(prev_des, curr_des, k=2)

            good = []
            for p in matches:
                if len(p) > 1 and p[0].distance < 0.7*p[1].distance:
                    good.append(p[0])

            prev_pts = np.float32([ prev_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            curr_pts = np.float32([ curr_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            
            stb_M, _ = cv2.findHomography(curr_pts, prev_pts, cv2.RANSAC,5.0)
                
            curr_stabilized = cv2.warpPerspective(curr_img, self.hom_M@stb_M, (self.w, self.h))
            curr_stabilized = cv2.warpAffine(curr_stabilized, self.rot_M, (self.w, self.h), flags = cv2.WARP_INVERSE_MAP)
            # out = unrot @ tar @ est @ curr
            
            with open("vid_"+self.transform_fn, 'a') as infile:
                t = (np.linalg.inv(np.vstack((self.rot_M, (0,0,1))))
                     @ self.hom_M @ stb_M @ np.diag([1/self.scales[0], 1/self.scales[0], 1]))
                infile.write(str(list(np.ravel(t))))
                if (i != self.frame_fin-1):
                    infile.write("\n")
            
            with open("rwc_"+self.transform_fn, 'a') as infile:
                t = (self.ref_M @ np.diag([1/self.scales[1], 1/self.scales[1], 1])
                     @ np.linalg.inv(np.vstack((self.rot_M, (0,0,1))))
                     @ self.hom_M @ stb_M @ np.diag([1/self.scales[0], 1/self.scales[0], 1]))
                infile.write(str(list(np.ravel(t))))
                if (i != self.frame_fin-1):
                    infile.write("\n")
                
            
            ##########################################################################################
            # apply the transforms to center of bounding boxes (x, y), then draw them
            
            # convert to numpy array
            coordinates = np.array(coordinates)[:,0,:2]

            homogeneous_coordinates = np.hstack((coordinates, np.ones((len(coordinates), 1))))

            # Combine the homography matrix and the stable matrix
            # combined_matrix = np.dot(self.hom_M, stb_M)
            sc1_M = np.diag([1/self.scales[0], 1/self.scales[0], 1])
            combined_matrix = self.hom_M @ stb_M @ sc1_M

            # Apply the combined transformation by matrix multiplication
            transformed_coordinates = np.dot(homogeneous_coordinates, combined_matrix.T)
            
            if i==0:
                print(coordinates.shape)
                print(homogeneous_coordinates.shape)
                print(transformed_coordinates.shape)

            # Apply the rotation transformation by matrix multiplication
            self.drt_M = np.linalg.inv(np.vstack(self.rot_M, np.ones(3)))
            transformed_coordinates = np.dot(transformed_coordinates, self.drt_M.T)

            # Extract the transformed (x, y) coordinates
            transformed_coordinates = transformed_coordinates[:, :2]
            print(transformed_coordinates.shape)

   
            for i in range(len(coordinates)):

                x, y = transformed_coordinates[i]
                cv2.circle(curr_stabilized, (int(x), int(y)), 5, (255, 0, 0), -1)
                cv2.putText(curr_stabilized, f"{int(ids[i])} ", (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
                cv2.putText(curr_stabilized, f"{int(cls[i])} ", (int(x), int(y + 25)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

            ##########################################################################################

            mask = (curr_stabilized != [0,0,0])
            targ = self.im_tar.copy()[:,:,2::-1]
            targ[mask] = curr_stabilized[mask]
            self.queue.put(targ)                

        self.queue.put(None)
        self.cap.release()