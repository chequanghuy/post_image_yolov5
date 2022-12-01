import io
import json                    
import base64                  
import logging             
import numpy as np
from PIL import Image
import time
import os
from flask import Flask, request, abort
import cv2
import json
import queue
  
import threading
import sys
sys.path.append('/home/huycq/yolov5')
sys.path.append('/home/huycq/yolov5/post_image_yolov5/ByteTrack')  # add ROOT to PATH
from yolox.utils.visualize import plot_tracking
app = Flask(__name__)          
app.logger.setLevel(logging.DEBUG)
  
receive_buffer=queue.Queue(maxsize=5)
  
@app.route("/test", methods=['POST'])
def test_method():         
    # print(request.json)      
    if not request.json or 'image' not in request.json: 
        abort(400)
    im_b64 = request.json['image']
    online_tlwhs = request.json['detection']
    online_ids = request.json['ids']
    # time=request.json['time']
    time_=time.time()
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))
    jpg_as_np = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    img_arr = np.asarray(img)  
    print(img_arr.shape) 
    if receive_buffer.full():
        receive_buffer.get()
    receive_buffer.put((img_arr,online_tlwhs,online_ids))
    # img = plot_tracking(
    #             img_arr, [np.array(d) for d in online_tlwhs], online_ids, frame_id=0, fps=1. / 1)
    # name ="{}/img_from_client/{}.jpg".format(os.getcwd(),time_)
    # dictionary = {
    #     "image dir": name,
    #     "bouding box": object_detection,
    #     "time": time_
    # }
    # json_object = json.dumps(dictionary, indent=3)
    # with open("sample.json", "a") as outfile:
    #     outfile.write(json_object)
    # cv2.imwrite(name,img)

    # result_dict = {'output': 'output_key'}
    return "Done"
  
  
def run_server_api():
    app.run(host='0.0.0.0', port=8080)

class Show(object):
    def __init__(self):
        self.buffer = queue.Queue(maxsize=5)
        self._running = True 
        self.show_thread=threading.Thread(target=self.show)
    def show(self):
        while True:
            if self.buffer.qsize()>0:
                img, online_tlwhs, online_ids=receive_buffer.get()
                if len(online_ids)>0:
                    img = plot_tracking(
                    img, [np.array(d) for d in online_tlwhs], online_ids, frame_id=0, fps=1. / 1)
                cv2.imshow("show",img)
                cv2.waitKey(1)
  
if __name__ == "__main__":     
    show = Show()
    show.buffer=receive_buffer
    show.show_thread.start()
    
    run_server_api()
    