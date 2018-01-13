
import cv2
import numpy as np

webcam = cv2.VideoCapture(0)

while (True):
	ret, frame = webcam.read()

	cv2.imshow('frame', frame)

	key = cv2.waitKey(10)
	if key == 27: # exit on ESC
		break

cv2.destroyWindow('frame')
webcam.release()
