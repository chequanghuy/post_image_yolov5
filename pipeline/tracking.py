from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from yolox.utils.visualize import plot_tracking
import threading
import queue
import numpy as np

class YoloTrack(object):
    def __init__(self,track_thresh=0.5,track_buffer=30,match_thresh=0.8): 
        # self.track_thresh=track_thresh
        # self.track_buffer=track_buffer
        # self.match_thresh=match_thresh
        self.aspect_ratio_thresh=1.6
        self.frame_id=0
        self.min_box_area=10
        self.tracker=BYTETracker(track_thresh,match_thresh,track_buffer)
        self.c_postprocess  = None 
        self.postprocess_buffer    = queue.Queue(maxsize=5)
        
        self.trackprocess_thread = threading.Thread(target=self.track_loop,daemon=False)
        
        self.trackprocess_buffer   = queue.Queue(maxsize=5)
        
        self._running = False 
        self.c_trackprocess  = threading.Condition()
        self.track4pre=threading.Condition()


        
    def track_loop(self):
        while True:
            with self.c_postprocess:
                self.c_postprocess.wait()

            img_resized , outputs = self.postprocess_buffer.get()
            online_im,online_tlwhs,online_ids=self.tracking_process(outputs,img_resized)
            
                
            if self.trackprocess_buffer.full():
                self.trackprocess_buffer.get()
            self.trackprocess_buffer.put((online_im,online_tlwhs,online_ids)) 
        
            with self.c_trackprocess:
                if (self.trackprocess_buffer.qsize() >= 1):
                    self.c_trackprocess.notifyAll()
            with self.track4pre:
                if (self.trackprocess_buffer.qsize() <=5):
                    self.track4pre.notifyAll()

    def tracking_process(self,outputs,img_resized):
        if outputs is not None:
            online_targets = self.tracker.update(np.array(outputs), [img_resized.shape[0], img_resized.shape[1]], img_resized.shape[:2])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > self.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            online_im = img_resized
            # timer.toc()
            # online_im = plot_tracking(
            #     img_resized, online_tlwhs, online_ids, frame_id=self.frame_id + 1, fps=1. / 1
            # )
        else:
            # timer.toc()
            online_im = img_resized
        self.frame_id+=1 
        return online_im,online_tlwhs,online_ids
        # return im0