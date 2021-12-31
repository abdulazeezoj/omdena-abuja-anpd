# import the necessary packages
import time

import cv2
from imutils.video import FileVideoStream, VideoStream


class ANPD():

    def __init__(self, configPath, weightsPath, namesPath,
                 confThreshold, nmsThreshold):
        self.configPath = configPath
        self.weightsPath = weightsPath
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.namesPath = namesPath

        print("[INFO] loading ANPD...")
        net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(scale=1/255, size=(416, 416), swapRB=True)

        print("[INFO] loading names...")
        with open(self.namesPath, 'r') as f:
            self.classNames = f.read().splitlines()

    def detect(self, frame):
        start = time.time()

        print("[INFO] detecting plate(s)...")
        classIds, scores, boxes = self.model.detect(
            frame, self.confThreshold, self.nmsThreshold)

        end = time.time()
        print("[INFO] ANPD took {:.6f} seconds".format(end - start))

        return classIds, scores, boxes

    def draw_bbox(self, frame, detection):

        classIds, scores, boxes = detection

        for (classId, score, box) in zip(classIds, scores, boxes):
            x, y, w, h = box
            bb_color = (0, 255, 255)
            text_color = (0, 0, 0)

            text = '%s: %.2f' % (self.classNames[classId], score)
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
        detection = self.detect(image)

        # render detection on the image
        result = self.draw_bbox(image, detection)

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
                detection = self.detect(frame)

                # render detection on the frame
                result = self.draw_bbox(frame, detection)

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
