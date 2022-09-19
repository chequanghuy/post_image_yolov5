import argparse
import os
import sys
from pathlib import Path

sys.path.append('/home/huycq/yolov5')  # add ROOT to PATH
from preprocess import YoloPre
from postprocess import YoloPost
from inference import YoloInfer
import cv2


pre=YoloPre(source='/home/huycq/yolov5/test_video.mp4')
pre._running=True
infer=YoloInfer()
post=YoloPost()

infer.preprocessed_buffer=pre.preprocessed_buffer 
infer.c_preprocessed=pre.c_preprocessed


post.c_inference=infer.c_inference
post.inference_buffer=infer.inference_buffer

pre.preprocessed_thread.start()
infer.inference_thread.start()
post.postprocess_thread.start()

while pre._running:
    # print("pre.preprocessed_buffer :",pre.preprocessed_buffer.qsize())
    # print("infer.c_inference :",infer.inference_buffer.qsize())
    # print("post.postprocess_buffer :",post.postprocess_buffer.qsize())
    if post.postprocess_buffer.qsize() >= 1:
        img,img0=post.postprocess_buffer.get()
        cv2.imshow('img',img)
        cv2.waitKey(1)
cv2.destroyAllWindows()


