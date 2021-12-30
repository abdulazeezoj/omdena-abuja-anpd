# import the necessary packages
import time
from typing import Any, List, Tuple
import cv2


def load_anpd(configPath: str, weightsPath: str) -> Any:
    """
    Load ANPD model.
    """
    print("[INFO] loading YOLO from disk...")

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1/255, size=(416, 416), swapRB=True)

    return model


def detect(model: Any, frame: Any,
           confThreshold: float, nmsThreshold: float) -> Tuple[Any, Any, Any]:
    """
    Detect number plate(s)
    """
    start = time.time()

    print("[INFO] detecting plate(s)...")
    classIds, scores, boxes = model.detect(frame, confThreshold, nmsThreshold)

    end = time.time()
    print("[INFO] ANPD took {:.6f} seconds".format(end - start))

    return classIds, scores, boxes


def draw_bbox(frame: Any, classNames: List[str],
              detection: Tuple[Any, Any, Any]) -> Any:
    """
    Draw detection bounding box.
    """

    classIds, scores, boxes = detection

    for (classId, score, box) in zip(classIds, scores, boxes):
        x, y, w, h = box
        bb_color = (0, 255, 255)
        text_color = (0, 0, 0)

        text = '%s: %.2f' % (classNames[classId], score)
        text_bb, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        frame = cv2.rectangle(frame, (x, y), (x + w, y + h),
                              color=bb_color, thickness=2)

        frame = cv2.rectangle(frame, (x - 1, y - (text_bb[1] + 5)),
                              (x + text_bb[0], y),
                              color=bb_color, thickness=-1)

        frame = cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color=text_color, thickness=1)

    return frame
