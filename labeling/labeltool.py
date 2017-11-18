import cv2
import sys
import os
import shutil


INPUT_FOLDER = "input/"
OUTPUT_FOLDER = "output/"

emotions = {
	"Anger" : 0,
	"Contempt" : 0,
	"Fear" : 0,
	"Disgust" : 0,
	"Happiness" : 0,
	"Sadness" : 0,
	"Surprise" : 0
}

keybinds = {}


def DeleteDirContents(dir):
	for f in os.listdir(dir):
		path = os.path.join(dir, f)
		try:
			if os.path.isfile(path):
				os.unlink(path)
		except Exception as e:
			print(e)

def ListFiles(dir):
	return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

def Init():
	global keybinds

	availableEmotions = list(emotions.keys())

	for i in range (1, len(emotions) + 1):
		keybinds[str(i)] = availableEmotions[i - 1]
		print(str(i) + " - " + keybinds[str(i)])

	print("Q - Quit")

	DeleteDirContents(OUTPUT_FOLDER)

def main():
	
	Init()

	for f in ListFiles(INPUT_FOLDER):
		crtImg = cv2.imread(INPUT_FOLDER + "/" + f)
		cv2.imshow("Labeling Utility", crtImg)

		validKeyBinds = [str(x) for x in keybinds.keys()]

		while True:
			key = chr(cv2.waitKey())

			if key in validKeyBinds:
				cv2.imwrite(OUTPUT_FOLDER + "/" + keybinds[key] + "_" + str(emotions[keybinds[key]]) + ".png", crtImg)
				emotions[keybinds[key]] = emotions[keybinds[key]] + 1
				break

			if key in ['q', 'Q']:
				sys.exit()
		


if __name__ == "__main__":
	main()
