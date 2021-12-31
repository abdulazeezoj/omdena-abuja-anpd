"""Omdena Abuja ANPD module"""

# import the necessary packages
import time as time

import cv2 as cv2
import easyocr
import imutils
from imutils.video import FileVideoStream, VideoStream


class ANPD():

    def __init__(self, anpdPath, confThreshold, nmsThreshold):
        self.__configPath = f"{anpdPath}anpd.cfg"
        self.__weightsPath = f"{anpdPath}out/anpd_best.weights"
        self.__namesPath = f"{anpdPath}anpd.names"
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold

        print("[INFO] loading ANPD...")
        self.__detector = cv2.dnn_DetectionModel(cv2.dnn.readNetFromDarknet(
            self.__configPath,
            self.__weightsPath))
        self.__detector.setInputParams(
            scale=1/255, size=(416, 416), swapRB=True)

        with open(self.__namesPath, 'r') as f:
            self.__classNames = f.read().splitlines()

        self.__reader = easyocr.Reader(['en'])

    def __detect(self, frame):
        start = time.time()

        print("[INFO] detecting plate(s)...")
        classIds, scores, boxes = self.__detector.detect(
            frame, self.confThreshold, self.nmsThreshold)

        end = time.time()
        print("[INFO] ANPD took {:.6f} seconds".format(end - start))

        return classIds, scores, boxes

    def __read(self, frame, detection):
        classIds, scores, boxes = detection
        numbers = []

        start = time.time()

        for box in boxes:
            x, y, w, h = box

            number = self.__reader.readtext(frame[y:y+h, x:x+w], detail=0)
            numbers.append(number[0].upper())

        end = time.time()
        print("[INFO] ANPR took {:.6f} seconds".format(end - start))

        return classIds, scores, boxes, numbers

    def __render(self, frame, detection):

        classIds, scores, boxes, numbers = detection

        for (classId, score, box, number) in zip(classIds, scores, boxes, numbers):
            x, y, w, h = box
            bb_color = (0, 255, 255)
            text_color = (0, 0, 0)

            text = '{} | {}[{:.0%}]'.format(number, self.__classNames[classId], score)
            text_bb, _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            frame = cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  color=bb_color, thickness=2)

            frame = cv2.rectangle(frame, (x - 1, y - (text_bb[1] + 5)),
                                  (x + text_bb[0], y),
                                  color=bb_color, thickness=-1)

            frame = cv2.putText(frame, text, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color=text_color, thickness=1)

        return frame

    def detectImage(self, imagePath):
        """
        Detect number plate in image.
        """

        # load our input image
        image = cv2.imread(imagePath)

        # detect number plate and show timing information
        detection = self.__detect(image)
        detection = self.__read(image, detection)

        # render detection on the image
        result = self.__render(image, detection)

        # show the output image
        cv2.imshow("Output", result)
        cv2.waitKey(0)

    def detectVideo(self, streamPath, streamType):

        # initialize the video stream and allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        if streamType == "video":
            stream = FileVideoStream(streamPath).start()
        elif streamType == "cam":
            stream = VideoStream(int(streamPath)).start()

        time.sleep(2.0)

        # loop over the frames from the video stream
        while True:
            frame = stream.read()

            if frame is not None:
                # detect number plate and show timing information
                detection = self.__detect(frame)
                detection = self.__read(frame, detection)

                # render detection on the frame
                result = self.__render(frame, detection)

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
