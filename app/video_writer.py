import threading
import cv2
import os
import queue
import tqdm
import numpy as np


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
        
        print('fps, n_frames, w, h: ', self.fps, self.n_frames, self.w, self.h)
        
        orb = cv2.ORB_create()
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        prev_kp, prev_des = orb.detectAndCompute(prev_img, mask=None)
        
        for i in tqdm(range(self.frame_ini,self.frame_fin)):
            success, curr_img = self.cap.read()
            if not success:
                break
            
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
            
            with open(self.transform_fn, 'a') as infile:
                t = (self.ref_M @ np.diag([1/self.scales[1], 1/self.scales[1], 1])
                     @ np.linalg.inv(np.vstack((self.rot_M, (0,0,1))))
                     @ self.hom_M @ stb_M @ np.diag([1/self.scales[0], 1/self.scales[0], 1]))
                infile.write(str(list(np.ravel(t))) + "\n")
            
            mask = (curr_stabilized != [0,0,0])
            targ = self.im_tar.copy()[:,:,2::-1]
            targ[mask] = curr_stabilized[mask]
            self.queue.put(targ)                

        self.queue.put(None)
        self.cap.release()