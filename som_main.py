
import file_utils
import som_classifier
import cv2
import numpy as np
import math
import datetime

from utils import InfiniteTimer
from image_crop import GetFaces
import read_and_test_model as read_model

# Available input modes: folder, webcam
# If folder is chosen, an input folder must be provided ($INPUT_FOLDER)
# If webcam is chosen, a device ID must be provided ($WEBCAM_ID)
INPUT_MODE = "webcam"

# Available output modes: folder, som
# If folder is chosen, an output folder must be provided ($OUTPUT_FOLDER)
# If som is chosen, this module will attempt to run the classifier on all images 
OUTPUT_MODE = "folder"

WEBCAM_ID = 0

INPUT_FOLDER = "input/"
OUTPUT_FOLDER = "output/"

# Initialize the neural network and load data model
model = read_model.initialize_network()
model = read_model.load_model(model)

SAMPLE_PERIOD_SECONDS = 5

def preprocess(frame):
	faces = GetFaces(frame, (128, 128), True)
	if len(faces) < 1:
		print("Could not find a face in this frame.`")
		return None

	im = np.asarray(faces[0], dtype='float64') / 256.
	im = read_model.convert_2D_to_3D(im)

	return im

def model_output(frame, res):
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(frame, "Anger: " + str(float(res[0][0]) * 100) + " %", (10, 30), font, 1, (255,255,0), 2, cv2.LINE_AA)
	cv2.putText(frame, "Disgust: " + str(float(res[0][1]) * 100) + " %", (10, 70), font, 1, (255,255,0), 2, cv2.LINE_AA)
	cv2.putText(frame, "Happiness: " + str(float(res[0][3]) * 100) + " %", (10, 110), font, 1, (255,255,0), 2, cv2.LINE_AA)
	cv2.putText(frame, "Sadness: " + str(float(res[0][5]) * 100) + " %", (10, 150), font, 1, (255,255,0), 2, cv2.LINE_AA)
	cv2.putText(frame, "Surprise: " + str(float(res[0][6]) * 100) + " %", (10, 190), font, 1, (255,255,0), 2, cv2.LINE_AA)


def sampleFolder():

	for inputFile in file_utils.ListFiles(INPUT_FOLDER):
			print("Current file: " + inputFile)
			videoCapture = cv2.VideoCapture(INPUT_FOLDER + "/" + inputFile)

			ret, frame = videoCapture.read()
			if ret:
				outputCurrentFrame()
			else:
				print("Could not open source")

			sampleRate = math.ceil(videoCapture.get(cv2.CAP_PROP_FPS)) * SAMPLE_PERIOD_SECONDS
			
			crtFrameCount = 0
			while ret == True:
				cv2.imshow('frame', frame)		
				if crtFrameCount % sampleRate == 0:
					if OUTPUT_MODE == "folder":
						cv2.imwrite(OUTPUT_FOLDER + "/" + inputFile.split(".")[0] + "_" + str(crtFrameCount) + ".png", frame)		
					elif OUTPUT_MODE == "som":
						# TODO run classifier
						pass
					else:
						print("Invalid output mode")

				crtFrameCount += 1
				ret, frame = videoCapture.read()

			videoCapture.release()


CURRENT_WEBCAM_FRAME = None

def outputCurrentFrame():
	if not CURRENT_WEBCAM_FRAME is None:
		frame = preprocess(CURRENT_WEBCAM_FRAME)

		if frame is None:
			return

		res = read_model.predict([frame], model)
		model_output(CURRENT_WEBCAM_FRAME, res)

		#cv2.imshow('frame', frame)
		cv2.imwrite(OUTPUT_FOLDER + "/" + str(datetime.datetime.now()).split(".")[0] + ".png", CURRENT_WEBCAM_FRAME)
		# if OUTPUT_MODE == "folder":
		# 	cv2.imwrite(OUTPUT_FOLDER + "/" + str(datetime.datetime.now()).split(".")[0] + ".png", CURRENT_WEBCAM_FRAME)
		# elif OUTPUT_MODE == "som":
		# 	# TODO run classifier
		# 	pass
		# else:
		# 	print("Invalid output mode")

# Takes a snapshot every X seconds
# Only works correctly with real-time feeds (AKA webcams)
def sampleWebcam():
	global CURRENT_WEBCAM_FRAME

	videoCapture = cv2.VideoCapture(WEBCAM_ID)
	t = InfiniteTimer(SAMPLE_PERIOD_SECONDS, outputCurrentFrame)
	t.start()

	ret, CURRENT_WEBCAM_FRAME = videoCapture.read()
	if ret:
		outputCurrentFrame()
	else:
		print("Could not open source")

	while ret == True:		
		ret, CURRENT_WEBCAM_FRAME = videoCapture.read()

	videoCapture.release()
	t.cancel()


if __name__ == "__main__":

	if INPUT_MODE is "folder":
		sampleFolder()
	elif INPUT_MODE is "webcam":
		sampleWebcam()
	else:
		print ("Invalid input mode")
