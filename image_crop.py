import numpy as np
import cv2

import file_utils

HAAR_FOLDER = "haarcascades/"
DEBUG_FOLDER = 'debug/'
INPUT_FOLDER = DEBUG_FOLDER + 'input/'
OUTPUT_FOLDER = DEBUG_FOLDER + 'output/'

# Can add more here
cascades = [cv2.CascadeClassifier(HAAR_FOLDER + 'haarcascade_frontalface_default.xml'), cv2.CascadeClassifier(HAAR_FOLDER + 'haarcascade_profileface.xml')]

# Returns a list of faces detected in @inputImg
# Faces are resized to @outputSize
# if @returnGrayscale is true,  the faces will be in grayscale
def GetFaces(inputImg, outputSize, returnGrayscale):
	
	result = []
	grayImage = cv2.cvtColor(inputImg, cv2.COLOR_BGR2GRAY)

	for cascade in cascades:
		faces = cascade.detectMultiScale(grayImage, 1.3, 5)

		for (x, y, w, h) in faces:
			result.append(cv2.resize((grayImage if returnGrayscale else inputImg)[y : y + h, x : x + w], outputSize, interpolation = cv2.INTER_LANCZOS4))

		if len(faces) > 0:
			break

	return result


# def main():	

# 	for image in file_utils.ListFiles(INPUT_FOLDER):
# 		cnt = 0
# 		for face in GetFaces(cv2.imread(INPUT_FOLDER + image), (256, 256), True):			
# 			cv2.imwrite(OUTPUT_FOLDER + image.split('.')[0] + '_face' + str(cnt) + '.' + image.split('.')[-1], face)
# 			cnt += 1

# if __name__ == "__main__":
# 	main()