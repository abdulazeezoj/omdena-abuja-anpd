"""Omdena Abuja ANPD module"""

# import the necessary packages
import time as time

import cv2 as cv2
import easyocr
from imutils.video import FileVideoStream, VideoStream


class ANPD():

    def __init__(self, anpdPath, confThreshold, nmsThreshold):
        self.__detectorConfig = f"{anpdPath}/detector/anpd.cfg"
        self.__detectorWeights = f"{anpdPath}/detector/anpd_best.weights"
        self.__detectorNames = f"{anpdPath}/detector/anpd.names"
        self.__readerPath = f"{anpdPath}/reader/"

        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold

        print("[INFO] loading ANPD...")
        self.__detector = cv2.dnn_DetectionModel(cv2.dnn.readNetFromDarknet(
            self.__detectorConfig,
            self.__detectorWeights))
        self.__detector.setInputParams(
            scale=1/255, size=(416, 416), swapRB=True)

        with open(self.__detectorNames, 'r') as f:
            self.__detectorNames = f.read().splitlines()

        self.__reader = easyocr.Reader(['en'], False, self.__readerPath,
                                       detector=False, verbose=False)
        self.__readerChars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890'

    def __detect(self, frame):
        start = time.time()

        print("[INFO] detecting plate(s)...")
        nameIds, scores, boxes = self.__detector.detect(
            frame, self.confThreshold, self.nmsThreshold)

        end = time.time()
        print("[INFO] ANPD took {:.6f} seconds".format(end - start))

        return nameIds, scores, boxes

    def __clean(self, plate):
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        return gray

    def __read(self, frame, detection):
        nameIds, scores, boxes = detection
        texts = []

        start = time.time()

        for box in boxes:
            x, y, w, h = box
            plate = self.__clean(frame[y:y+h, x:x+w])

            text = self.__reader.recognize(plate, detail=0,
                                           allowlist=self.__readerChars)

            if len(text) > 0:
                texts.append(text[0].upper())
            else:
                texts.append('')

        end = time.time()
        print("[INFO] ANPR took {:.6f} seconds".format(end - start))

        return nameIds, scores, boxes, texts

    def __render(self, frame, detection):

        nameIds, scores, boxes, texts = detection

        for nameId, score, box, text in zip(nameIds, scores, boxes, texts):
            x, y, w, h = box
            bb_color = (0, 255, 255)
            text_color = (0, 0, 0)

            text = '{} | {}[{:.0%}]'.format(
                text, self.__detectorNames[nameId], score)
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
