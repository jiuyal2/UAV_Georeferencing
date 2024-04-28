import threading
import cv2

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