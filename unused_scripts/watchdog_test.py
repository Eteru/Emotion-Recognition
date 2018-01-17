import sys
import time
import cv2
import json
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


from utils import InfiniteTimer
from image_crop import GetFaces
import read_and_test_model as read_model


class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory == False:
            fname = event.src_path.split("/")[-1]
            if fname.startswith(".") == True:
                return
            
            res = compute(event.src_path)
            write_result(res, fname.split(".")[0] + ".json")

    def compute(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = preprocess(img)

        if img is None:
            return

        res = read_model.predict([frame], model)

        return res

    def write_result(self, res, path):
        print(path, res)
    
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
observer.schedule(event_handler, "/home/ciprian/test", recursive=True)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()