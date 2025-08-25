# analyzing model:
import numpy as np
import imutils
import cv2
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours

# function to show arrat of images:
def show_images(images):
	for i, img in enumerate(images):
		cv2.imshow("image_" + str(i), img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# include the path to the image you want to analyze:
img_path = "obj2.jpeg"

# Read image and preprocess
image = cv2.imread(img_path)

# resizing:

