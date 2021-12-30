import argparse
import time

import cv2
from imutils.video import FileVideoStream, VideoStream

from src import anpd


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True,
                help="what to run detection on")
ap.add_argument("-t", "--type", default='image',
                help="type of what to run detection on; image / cam / video")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-th", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())


# load ANPD
model = anpd.load_anpd("anpd/anpd.cfg",
                       "anpd/out/anpd_best.weights")

# load class names
with open("anpd/anpd.names", 'r') as f:
    classNames = f.read().splitlines()


def detectImage(src):
    """
    Detect number plate in image.
    """

    # load our input image
    image = cv2.imread(args["source"])

    # load class names
    with open("anpd/anpd.names", 'r') as f:
        classNames = f.read().splitlines()

    # load ANPD
    model = anpd.load_anpd("anpd/anpd.cfg",
                           "anpd/out/anpd_best.weights")

    # detect number plate and show timing information
    detection = anpd.detect(
        model, image, args["confidence"], args["threshold"])

    # render detection on the image
    result = anpd.draw_bbox(image, classNames, detection)

    # show the output image
    cv2.imshow("Output", result)
    cv2.waitKey(0)


def detectVideo(src, src_type):

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    if src_type == "video":
        stream = FileVideoStream(src).start()
    elif src_type == "cam":
        stream = VideoStream(int(src)).start()

    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        frame = stream.read()

        if frame is not None:
            # detect number plate and show timing information
            detection = anpd.detect(model, frame, args["confidence"],
                                    args["threshold"])

            # render detection on the frame
            result = anpd.draw_bbox(frame, classNames, detection)

            # display result
            cv2.imshow('Result', result)
        else:
            print("[INFO] end of video stream ...")
            break

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            print("[INFO] video stream closed ...")
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    stream.stop()


src_type = args["type"]
src = args["source"]

if src_type == "image":
    detectImage(src)
elif src_type == "video":
    detectVideo(src, src_type)
elif src_type == "cam":
    detectVideo(src, src_type)
else:
    print("[INFO] source undetermined")
