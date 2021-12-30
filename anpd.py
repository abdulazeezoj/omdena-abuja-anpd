# import the necessary packages
import argparse
import time
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())


# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])

# load the objects class names our YOLO model was trained on
with open("models/anpd.names", 'r') as f:
    classNames = f.read().splitlines()

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("anpd/anpd.cfg",
                                 "anpd/anpd_best.weights")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1/255, size=(416, 416), swapRB=True)


# start detecting object and show timing information on YOLO
start = time.time()

print("[INFO] detecting object(s)...")
classIds, scores, boxes = model.detect(
    image, confThreshold=args["confidence"], nmsThreshold=args["threshold"])

end = time.time()
print("[INFO] YOLO took {:.6f} seconds".format(end - start))


# initialize detected bounding boxes, confidences, and
# class IDs, respectively
for (classId, score, box) in zip(classIds, scores, boxes):
    cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                  color=(0, 255, 0), thickness=2)

    text = '%s: %.2f' % (classNames[classId], score)
    cv2.putText(image, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color=(0, 255, 0), thickness=2)


# display result
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
