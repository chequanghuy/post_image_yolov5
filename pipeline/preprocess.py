

import argparse
import os
import sys
from pathlib import Path
import cv2
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
        self.preprocessed_buffer   = queue.Queue() #maxsize=5
        self._running = False 
        self.stride=64
    def pre_loop(self):
        while True:
            img0=self.image_loader()
            img0,img=self.preprocess(img0)
            if self.preprocessed_buffer.full():
                self.preprocessed_buffer.get()
            self.preprocessed_buffer.put((img0,img)) 
            with self.c_preprocessed:
                if (self.preprocessed_buffer.qsize() >= 1):
                    self.c_preprocessed.notifyAll()
    
    def preprocess(self,img0):
        img = letterbox(img0, self.img_size, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1] 
        img = np.ascontiguousarray(img)
        return img0,img
    
    def image_loader(self):
        # print("here")
        ret, frame = self.cap.read()
        if ret == True:
            return frame
        self._running=False


