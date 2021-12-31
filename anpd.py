import argparse

from anpd.anpd import ANPD

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
model = ANPD("anpd/anpd.cfg", "anpd/out/anpd_best.weights",
             "anpd/anpd.names", args["confidence"], args["threshold"])

src_type = args["type"]
src = args["source"]

if src_type == "image":
    model.detectImage(src)
elif src_type == "video":
    model.detectVideo(src, src_type)
elif src_type == "cam":
    model.detectVideo(src, src_type)
else:
    print("[INFO] source undetermined")
