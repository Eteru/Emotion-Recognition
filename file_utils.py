import sys
import os
import shutil

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