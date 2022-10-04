from models.experimental import attempt_load
from pathlib import Path
import cv2
import numpy as np
from utils.augmentations import letterbox
import torch
import threading
import queue
from utils.torch_utils import select_device

class YoloInfer(object):
    def __init__(self, weights='/home/huycq/yolov5/s_640.pt', device='0'): 

        self.device = select_device(device)
        self.model = attempt_load(weights, map_location=self.device)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # self.stride = int(self.model.stride.max())
        self.half=False
        self.c_preprocessed  = None 
        self.preprocessed_buffer   = queue.Queue()

        self.inference_thread = threading.Thread(target=self.infer_loop,daemon=False)
        self.c_inference  = threading.Condition() 
        self.inference_buffer   = queue.Queue()

        self._running = False 

    def infer_loop(self):
        while True:
            with self.c_preprocessed:
                self.c_preprocessed.wait()

            img_resized,img=self.preprocessed_buffer.get()

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None] 
            with torch.no_grad():
                pred = self.model(img, augment=False, visualize=False)[0]

            if self.inference_buffer.full():
                self.inference_buffer.get()
            self.inference_buffer.put((img_resized,pred)) 

            with self.c_inference:
                if (self.inference_buffer.qsize() >= 1):
                    self.c_inference.notifyAll()

