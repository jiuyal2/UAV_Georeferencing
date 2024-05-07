from scipy import signal
import numpy as np
import cv2
import os
import queue
from collections import deque
from src.detector.detection import DetectionModel
from tqdm import tqdm
from src.video_writer import VideoWriterThread
from src.georef_helper import resize_to_max_dim

class Georeference():
    def __init__(self, source:cv2.VideoCapture, target:np.ndarray,
                 rwc_M:np.ndarray, hom_M:np.ndarray, output_prefix:str=""):
        """
        Automatically stabilizes video. Capable of producing a georeferencing
        transform file or georeferenced video. In the future, this class will
        handle data analysis, like traffic heat maps and speed display.
        
        Parameters:
            source: A cv2.VideoCapture object containing the source aerial video.
            target: A numpy array containing the image data with shape (H, W, 3)
            rwc_M: A numpy array containing the transform from target image coordinates
                to real world coordinates under a specific projection.
            hom_M: A numpy array containing the transform from source video to target image coordinates.
            output_prefix: A prefix to add to the default video and text output filenames.
        """
        
        self.source = source
        self.target = target
        self.rwc_M = rwc_M
        self.hom_M = hom_M
        self.output_prefix = output_prefix
        self.output_dir = os.path.join(os.getcwd(), f"output/{self.output_prefix}")
        self.vid_out_filename = os.path.join(self.output_dir, "stb.mp4")
        self.txt_out_filename = os.path.join(self.output_dir, "tfs.txt")
        self.fps = source.get(cv2.CAP_PROP_FPS)
    
    def run(self, frame_ini:int, frame_fin:int, resolution:int = 1080):
        """
        Run the georeferencing object on the provided video and image.
        Save the results to {prefix}_stb.mp4 and {prefix}_tfs.txt.
        
        Parameters:
            frame_ini: Frame to use as initial frame for output video.
            frame_fin: Frame to use as final frame for output video.
            resolution: Optional maximum dimension of output video.
        """
        im_tar_sm, scale_factor = resize_to_max_dim(self.target, resolution, True)
        vid_h, vid_w, *_ = im_tar_sm.shape
        vidqueue = queue.Queue()
        
        video_writer_thread = VideoWriterThread(vidqueue, self.vid_out_filename,
                                                self.fps, vid_w, vid_h)
        video_writer_thread.start()
        self.source.set(cv2.CAP_PROP_POS_FRAMES, frame_ini)
        if frame_fin == -1:
            frame_fin = int(self.source.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_fin > int(self.source.get(cv2.CAP_PROP_FRAME_COUNT)):
            frame_fin = int(self.source.get(cv2.CAP_PROP_FRAME_COUNT))
        
        orb = cv2.ORB_create()
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        det_model = DetectionModel()

        success, base_img = self.source.read()
        assert success == True, f"Problems reading file: {self.vid_in_path}"
        base_kp, base_des = orb.detectAndCompute(base_img, mask=None)
        
        self.source.set(cv2.CAP_PROP_POS_FRAMES, frame_ini)
        
        kine_dict = dict()
        b,a = signal.butter(2, 0.5)

        for i in tqdm(range(frame_ini, frame_fin)):
            success, curr_img = self.source.read()
            if not success:
                break
                    
            #########################################################
            ##### FIND STB TRANSFORM, OUTPUT ########################
            #########################################################
            
            curr_kp, curr_des = orb.detectAndCompute(curr_img, None)
            matches = flann.knnMatch(base_des, curr_des, k=2)

            good = []
            for p in matches:
                if len(p) > 1 and p[0].distance < 0.7*p[1].distance:
                    good.append(p[0])

            base_pts = np.float32([ base_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            curr_pts = np.float32([ curr_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            
            stb_M, _ = cv2.findHomography(curr_pts, base_pts, cv2.RANSAC,5.0)
            
            with open(self.txt_out_filename, 'a') as infile:
                t = (self.rwc_M @ self.hom_M @ stb_M)
                infile.write(str(list(np.ravel(t))))
                if (i != frame_fin-1):
                    infile.write("\n")
            
            vid_M = np.diag((scale_factor, scale_factor, 1)) @ self.hom_M @ stb_M
            wor_M = self.rwc_M @ self.hom_M @ stb_M
            curr_stabilized = cv2.warpPerspective(curr_img, vid_M, (vid_w, vid_h))
                
            #########################################################
            ##### TRACKING OVERLAY ##################################
            #########################################################
            
            det_model.process_frame(curr_img)
            coordinates = det_model.get_coordinate()
            ids = det_model.get_id()
            cls = det_model.get_class()
            
            # apply the transforms to center of bounding boxes (x, y), 
            coordinates = np.squeeze(np.array(coordinates)[:,0,:2])
            homogeneous_coordinates = np.hstack((coordinates, np.ones((len(coordinates), 1))))
            video_coordinates = np.dot(homogeneous_coordinates, vid_M.T)
            world_coordinates = np.dot(homogeneous_coordinates, wor_M.T)
            video_coordinates /= video_coordinates[:, 2].reshape(-1,1)
            world_coordinates /= world_coordinates[:, 2].reshape(-1,1)
            
            # Extract the transformed (x, y) coordinates
            world_coordinates = world_coordinates[:, :2]
            transformed_coordinates = video_coordinates[:, :2]
            
            # draw coordinates on stabilized frame
            # print(kine_dict)
            for j in range(len(coordinates)):
                x, y = transformed_coordinates[j]
                wx, wy = world_coordinates[j]
                
                kine_list = kine_dict.get(int(ids[j]))
                if kine_list is None:
                    kine_list = [deque(maxlen=8), deque(maxlen=8), deque(maxlen=8)]
                    kine_dict[int(ids[j])] = kine_list
                ox, oy, of = kine_list
                ox.append(wx); oy.append(wy); of.append(i)
                if len(ox) > 2:
                    nx = signal.filtfilt(b,a,ox, padtype=None)
                    ny = signal.filtfilt(b,a,oy, padtype=None)
                    v = np.linalg.norm([nx[-1]-nx[-2], ny[-1]-ny[-2]])*self.fps/(of[-1]-of[-2])
                    cv2.putText(curr_stabilized, f"{round(v, 2)} ", (int(x+10), int(y+5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1)

                cv2.circle(curr_stabilized, (int(x), int(y)), 3, (0, 0, 255), -1)
                cv2.putText(curr_stabilized, f"{int(ids[j])} ", (int(x-10), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(curr_stabilized, f"{int(cls[j])} ", (int(x), int(y + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 0)

            mask = (curr_stabilized != [0,0,0])
            targ = im_tar_sm.copy()[:,:,2::-1]
            targ[mask] = curr_stabilized[mask]
            vidqueue.put(targ)

        vidqueue.put(None)