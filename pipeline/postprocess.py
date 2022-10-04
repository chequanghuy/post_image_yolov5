

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from utils.augmentations import letterbox
import torch
import threading
import queue
from utils.general import non_max_suppression
from utils.plots import Annotator, colors

class YoloPost(object):
    def __init__(self, conf_thres=0.25, iou_thres=0.45,max_det=1000): 

        
        self.c_inference  = None 
        self.inference_buffer    = queue.Queue(maxsize=5)
        
        self.postprocess_thread = threading.Thread(target=self.post_loop,daemon=False)
        
        self.postprocess_buffer   = queue.Queue(maxsize=5)
        
        self._running = False 

        self.c_postprocess=threading.Condition()

        self.c_postprocess4pre=threading.Condition()
        self.conf_thres=conf_thres
        self.iou_thres=iou_thres
        self.max_det=max_det
        self.visualize=True
        self.names=[
            'person',
            'motorbike',
            'bicycle',
            'face',
            'plate',
            'longplate',
            'car',
            'truck',
            'van',
            'bus',
            'bagac'
        ]
        
    def post_loop(self):
        while True:
            with self.c_inference:
                self.c_inference.wait()

            img_resized,pred=self.inference_buffer.get()
            results=self.postprocess(pred)
            
                
            if self.postprocess_buffer.full():
                self.postprocess_buffer.get()
            self.postprocess_buffer.put((img_resized,results)) 
        
            with self.c_postprocess:
                if (self.postprocess_buffer.qsize() >= 1):
                    self.c_postprocess.notifyAll()
            # with self.c_postprocess4pre:
            #     if (self.postprocess_buffer.qsize() <=5):
            #         self.c_postprocess4pre.notifyAll()

    def postprocess(self,pred):
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=self.max_det)
        results=[]
        for i, det in enumerate(pred):  # per image
            det=det.cpu().detach().numpy()
            # im0= im0s.copy()
            
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    if int(cls)==0:
                        results.append([*xyxy,conf])
                    # im0 = cv2.rectangle(im0, (int(xyxy[0]),int(xyxy[1])), (int(xyxy[2]),int(xyxy[3])), (255,0,0), 2)           
        return results
