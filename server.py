import io
import json                    
import base64                  
import logging             
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify, abort
import cv2
app = Flask(__name__)          
app.logger.setLevel(logging.DEBUG)
  
  
@app.route("/test", methods=['POST'])
def test_method():         
    # print(request.json)      
    if not request.json or 'image' not in request.json: 
        abort(400)
             
    # get the base64 encoded string
    im_b64 = request.json['image']
    object_detection = request.json['detection']
    print(object_detection)
    # time=request.json['time']
    # print(time)
    # convert it into bytes  
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    # convert bytes data to PIL Image object
    # jpg_original = base64.b64decode(img_bytes)
    jpg_as_np = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)

    # PIL image object to numpy array
    img_arr = np.asarray(img)   
    time="88888"  
    name =time.replace("/","-")
    print('img shape', img_arr.shape)
    cv2.imwrite(name+".jpg",img_arr)
    # process your img_arr here    
    
    # access other keys of json
    # print(request.json['other_key'])

    result_dict = {'output': 'output_key'}
    return result_dict
  
  
def run_server_api():
    app.run(host='0.0.0.0', port=8080)
  
  
if __name__ == "__main__":     
    run_server_api()