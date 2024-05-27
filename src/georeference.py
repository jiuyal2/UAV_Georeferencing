from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import cv2
import os
import queue
from collections import deque
from itertools import islice
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
        self.speed_map = np.zeros((1,1))
        self.volume_map = np.zeros((1,1))
        self.sample_img = None
    
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
        self.speed_map = np.zeros((vid_h, vid_w))
        self.volume_map = np.zeros((vid_h, vid_w))
        self.speed_overlay = np.zeros((vid_h, vid_w))
        
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
                x, y = transformed_coordinates[j] # video frame coordinates
                wx, wy = world_coordinates[j] # millions of meters, beware!
                
                kine_list = kine_dict.get(int(ids[j]))
                if kine_list is None:
                    kine_list = [deque(maxlen=8), deque(maxlen=8), deque(maxlen=8), deque(maxlen=20), deque(maxlen=20)]
                    kine_dict[int(ids[j])] = kine_list
                kx, ky, kf, vx, vy = kine_list
                kx.append(wx); ky.append(wy); kf.append(i); vx.append(x); vy.append(y)
                postmid = (len(kx)+1)//2
                premid = (len(kx))//2
                kx0 = np.mean(list(islice(kx, postmid))); kx1 = np.mean(list(islice(kx, premid, len(kx))))
                ky0 = np.mean(list(islice(ky, postmid))); ky1 = np.mean(list(islice(ky, premid, len(ky))))
                kf0 = np.mean(list(islice(kf, postmid))); kf1 = np.mean(list(islice(kf, premid, len(kf))))
                if kf0 == kf1:
                    v = 0
                else:
                    if abs(kx1-kx0) < 0.5:
                        kx1 = kx0
                    if abs(ky1-ky0) < 0.5:
                        ky1 = ky0
                    v = np.linalg.norm([kx1-kx0, ky1-ky0])*self.fps/(kf1-kf0)
                cv2.putText(curr_stabilized, f"{round(v, 2)} ", (int(x+10), int(y+5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,80), 1)
                self.speed_map[int(y), int(x)] += v
                self.volume_map[int(y), int(x)] += 1
                cv2.circle(curr_stabilized, (int(x), int(y)), 3, (0, 0, 255), -1)
                cv2.putText(curr_stabilized, f"{int(ids[j])} ", (int(x-10), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 255, 40), 1)
                cv2.putText(curr_stabilized, f"{int(cls[j])} ", (int(x), int(y + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 100, 255), 0)
                # if len(vx) > 1:
                #     curr_stabilized = cv2.polylines(curr_stabilized, [np.array(list(zip(vx,vy)), dtype=np.int32)], False, (255,255,0), 2)

            mask = (curr_stabilized != [0,0,0])
            targ = im_tar_sm.copy()[:,:,2::-1]
            targ[mask] = curr_stabilized[mask]
            vidqueue.put(targ)
            
            if i == int(0.5*(frame_ini + frame_fin)):
                self.sample_img = targ

        vidqueue.put(None)
        self.volume_map = self.blur(self.volume_map)
        self.speed_map = self.blur(self.speed_map)
        self.speed_map /= (frame_fin - frame_ini)
        
        mask = (self.speed_map != 0)
        targ = im_tar_sm.copy()[:,:,2::-1]
        
        cm = plt.get_cmap("inferno")
        mcm = cm(np.arange(cm.N))
        mcm[:,-1] = np.linspace(0,1,cm.N)**0.4
        mcm = col.ListedColormap(mcm)
        colored = (mcm(self.speed_map / np.max(self.speed_map))*255).astype(np.uint8)
        colomask = colored[mask]
        
        targ[mask] = colomask[:,-1][:,None]/255*colomask[:,:-1] + (255-colomask[:,-1][:,None])/255*targ[mask]
        
        self.speed_overlay = targ
        
    def blur(self, map):
        base_kernel = [1,1,2,3,4,5,5,5,5,5,5,5,5,3,1]
        kernel = np.convolve(base_kernel, base_kernel[::-1])
        kernel = np.clip(kernel, 0, 40)
        map = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 1, map)
        map = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, map)
        return map