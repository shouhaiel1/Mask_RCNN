import cv2
import argparse
import numpy as np
from visualize_cv2 import model, display_instances, class_names
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to our input video")
args = vars(ap.parse_args())
capture = cv2.VideoCapture(args["input"])
size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('masked_video.avi', codec, 25.0, size)

while(capture.isOpened()):
    ret, frame = capture.read()
    if ret:
        # add mask to frame
        results = model.detect([frame], verbose=0)
        r = results[0]

        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'] 
        )
            
        output.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
output.release()
cv2.destroyAllWindows()
