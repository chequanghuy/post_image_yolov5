import argparse
import os
import sys
from pathlib import Path
sys.path.append('/home/huycq/yolov5')
sys.path.append('/home/huycq/yolov5/post_image_yolov5/ByteTrack')  # add ROOT to PATH
from preprocess import YoloPre
from postprocess import YoloPost
from inference import YoloInfer
from tracking import YoloTrack
from client import Client
import cv2


pre=YoloPre(source='/home/huycq/yolov5/palace.mp4')
pre._running=True
infer=YoloInfer()
post=YoloPost()
track=YoloTrack()
send_egine=Client()

infer.preprocessed_buffer=pre.preprocessed_buffer 
infer.c_preprocessed=pre.c_preprocessed


post.c_inference=infer.c_inference
post.inference_buffer=infer.inference_buffer

track.c_postprocess=post.c_postprocess
track.postprocess_buffer=post.postprocess_buffer

send_egine.trackprocess_buffer=track.trackprocess_buffer
send_egine.c_track=track.c_trackprocess

pre.track4pre=track.track4pre

pre.preprocessed_thread.start()
infer.inference_thread.start()
post.postprocess_thread.start()
track.trackprocess_thread.start()
send_egine.send_data_thread.start()

# while pre._running:
#     with track.c_trackprocess:
#         track.c_trackprocess.wait()
#     if track.trackprocess_buffer.qsize() >= 1:
#         img=track.trackprocess_buffer.get()
#         cv2.imshow('img',img)
#         cv2.waitKey(10)
# cv2.destroyAllWindows()


