import os
import sys
import time
import cv2
import json
import datetime
import numpy as np

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from utils import InfiniteTimer
from image_crop import GetFaces
import read_and_test_model as read_model

model = read_model.initialize_network()
model = read_model.load_model(model)

print("Ready.")

class MyHandler(FileSystemEventHandler):
    """ No need for on modified for ths functionality
    def on_modified(self, event):
        print("on_modified: " + event.src_path)
        self.common_parse(event)
    """
    
    def on_created(self, event):
        print("on_created: " + event.src_path)
        self.wait_for_file(event.src_path)
        self.common_parse(event)
    
    def on_moved(self, event):
        print("on_moved: " + event.src_path)
        self.wait_for_file(event.src_path)
        self.common_parse(event)
    
    def wait_for_file(self, path):
        hsize = -1

        while hsize != os.path.getsize(path):
            hsize = os.path.getsize(path)
            time.sleep(1)

    def common_parse(self, event):
        if event.is_directory == False:
            fname = event.src_path.split("/")[-1]
            if fname.startswith(".") == True:
                return
            
            print(event.src_path)
            res = self.compute(event.src_path)
            if res is None:
                return

            self.write_result(res, fname.split(".")[0])
    

    def compute(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        img = self.preprocess(img)

        if img is None:
            return None

        res = read_model.predict([img], model)

        return res

    def write_result(self, res, id):
        #create json dict and write it
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        D = {
            "id" : id,
            "timestamp" : st,
            "emotions" : {
                "anger" : res[0][0],
                "disgust" : res[0][1],
                "happiness" : res[0][3],
                "sadness": res[0][5],
                "surprise" : res[0][6]
            }
        }

        with open(id + ".json", "w") as outfile:
            json.dump(D, outfile)
    
    def preprocess(self, img):
        faces = GetFaces(img, (128, 128), True)
        if len(faces) < 1:
            print("Could not find a face.`")
            return None

        im = np.asarray(faces[0], dtype='float64') / 256.
        im = read_model.convert_2D_to_3D(im)

        return im

event_handler = MyHandler()
observer = Observer()
observer.schedule(event_handler, "/home/ciprian/test", recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()