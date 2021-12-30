# import the necessary packages
import argparse
import time
import cv2
from src import anpd


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())


# load our input image
image = cv2.imread(args["image"])

# load class names
with open("anpd/anpd.names", 'r') as f:
    classNames = f.read().splitlines()

# load ANPD
model = anpd.load_anpd("anpd/anpd.cfg",
                       "anpd/out/anpd_best.weights")


# detect number plate and show timing information
detection = anpd.detect(model, image, args["confidence"], args["threshold"])

# render detection on the image
result = anpd.draw_bbox(image, classNames, detection)

# display result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
