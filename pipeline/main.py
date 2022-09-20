import argparse
import os
import sys
from pathlib import Path

sys.path.append('/home/tx2jp462/yolov5')  # add ROOT to PATH
from preprocess import YoloPre
from postprocess import YoloPost
from inference import YoloInfer
import cv2


pre=YoloPre(source='/home/tx2jp462/Downloads/test.mp4')
pre._running=True
infer=YoloInfer()
post=YoloPost()

infer.preprocessed_buffer=pre.preprocessed_buffer 
infer.c_preprocessed=pre.c_preprocessed


post.c_inference=infer.c_inference
post.inference_buffer=infer.inference_buffer

pre.c_postprocess4pre=post.c_postprocess4pre

pre.preprocessed_thread.start()
infer.inference_thread.start()
post.postprocess_thread.start()

while pre._running:
    with post.c_postprocess:
        post.c_postprocess.wait()
    if post.postprocess_buffer.qsize() >= 1:
        img=post.postprocess_buffer.get()
        cv2.imshow('img',img)
        cv2.waitKey(10)
cv2.destroyAllWindows()


