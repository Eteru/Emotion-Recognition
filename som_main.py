import file_utils
import som_classifier
import cv2
import math
import datetime

from utils import InfiniteTimer

# Available input modes: folder, webcam
# If folder is chosen, an input folder must be provided ($INPUT_FOLDER)
# If webcam is chosen, a device ID must be provided ($WEBCAM_ID)
INPUT_MODE = "folder"

# Available output modes: folder, som
# If folder is chosen, an output folder must be provided ($OUTPUT_FOLDER)
# If som is chosen, this module will attempt to run the classifier on all images 
OUTPUT_MODE = "folder"


# I think this is 0
# TODO Aprodu test it
WEBCAM_ID = "???"

INPUT_FOLDER = "input/"
OUTPUT_FOLDER = "output/"

SAMPLE_PERIOD_SECONDS = 5

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
		if OUTPUT_MODE == "folder":
			cv2.imwrite(OUTPUT_FOLDER + "/" + str(datetime.datetime.now()).split(".")[0] + ".png", CURRENT_WEBCAM_FRAME)
		elif OUTPUT_MODE == "som":
			# TODO run classifier
			pass
		else:
			print("Invalid output mode")

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
