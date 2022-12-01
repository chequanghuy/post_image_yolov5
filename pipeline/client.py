import base64
import json                    
import cv2
import requests
import threading
import queue
class Client(object):
    def __init__(self):
        self.headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        self.api='http://localhost:8080/test'
        self.send_data_thread = threading.Thread(target=self.send,daemon=False)
        self.c_track = None
        self.trackprocess_buffer = queue.Queue(maxsize=5)
        
        self._running = True 
    def send(self):
        while True:
            # print("check")
            with self.c_track:
                self.c_track.wait()
            # print("size :",self.trackprocess_buffer.qsize())
            image,online_tlwhs,online_ids=self.trackprocess_buffer.get()
            # print(image.shape)
            success, encoded_image = cv2.imencode('.png', image)
            im_bytes = encoded_image.tobytes()   
            im_b64 = base64.b64encode(im_bytes).decode("utf8")
            # print(type(online_tlwhs[]),type(online_ids))
            payload = json.dumps({"image": im_b64, "detection": [list(d) for d in online_tlwhs], "ids": online_ids})
            response = requests.post(self.api, data=payload, headers=self.headers)
            try:
                data = response.json()     
                print("try :",data)                
            except requests.exceptions.RequestException:
                print("except :",response.text)
            if not self._running:
                break