

import argparse
import os
import sys
from pathlib import Path
import cv2
import time
import numpy as np
from utils.augmentations import letterbox

import threading
import queue


class YoloPre(object):
    def __init__(self, source=''): 
        self.img_size = 640
        self.cap = cv2.VideoCapture(source)
        self.preprocessed_thread = threading.Thread(target=self.pre_loop,daemon=False)
        self.c_preprocessed  = threading.Condition()  
        self.preprocessed_buffer   = queue.Queue(maxsize=3) #maxsize=5
        self._running = False 
        self.stride=64
        self.delay=True
        self.c_postprocess4pre=threading.Condition()
    def pre_loop(self):
        while True:
            
                
            img0=self.image_loader()
            img_resized,img=self.preprocess(img0)
            if self.preprocessed_buffer.full():
                self.preprocessed_buffer.get()
            self.preprocessed_buffer.put((img_resized,img)) 
            with self.c_preprocessed:
                if (self.preprocessed_buffer.qsize() >= 1):
                    self.c_preprocessed.notifyAll()
            with self.c_postprocess4pre:
                self.c_postprocess4pre.wait()
            # if self.delay:
            #     time.sleep(10)
            #     self.delay=False
    def preprocess(self,img0):
        img_resized = letterbox(img0, self.img_size, stride=self.stride, auto=False)[0]
        img = img_resized.transpose((2, 0, 1))[::-1] 
        img = np.ascontiguousarray(img)
        return img_resized,img
    
    def image_loader(self):
        # print("here")
        ret, frame = self.cap.read()
        if ret == True:
            return frame
        self._running=False


