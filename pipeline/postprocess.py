

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
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import Annotator, colors

class YoloPost(object):
    def __init__(self, conf_thres=0.25, iou_thres=0.45,max_det=1000): 

        
        self.c_inference  = None 
        self.inference_buffer    = queue.Queue()
        
        self.postprocess_thread = threading.Thread(target=self.post_loop,daemon=False)
        
        self.postprocess_buffer   = queue.Queue()
        
        self._running = False 
        
        self.conf_thres=conf_thres
        self.iou_thres=iou_thres
        self.max_det=max_det
        self.visualize=True
        
    def post_loop(self):
        while True:
            with self.c_inference:
                self.c_inference.wait()

            img0,img,pred,self.names=self.inference_buffer.get()
            im0=self.postprocess(pred,img0,img)
            
                
            if self.postprocess_buffer.full():
                self.postprocess_buffer.get()
            self.postprocess_buffer.put((im0,img0)) 
        
        # with self.c_postprocess:
        #     if (self.postprocess_buffer.qsize() >= 1):
        #         self.c_postprocess.notifyAll()

    def postprocess(self,pred,im0s,img):
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=self.max_det)
        
        for i in range(len(pred)):
            pred[i]=np.rint(pred[i].cpu().detach().numpy())
        for i, det in enumerate(pred):  # per image
            im0= im0s.copy()
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    # xyxy=xyxy.cpu().detach().numpy()
                    print((xyxy[0],xyxy[1]), (xyxy[2],xyxy[3]))
                    im0 = cv2.rectangle(im0, (int(xyxy[0]),int(xyxy[1])), (int(xyxy[2]),int(xyxy[3])), (255,0,0), 2)
                    
                
        return im0
