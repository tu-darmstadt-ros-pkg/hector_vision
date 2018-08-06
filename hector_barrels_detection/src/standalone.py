#!/usr/bin/env python
import argparse
import cv2
import matplotlib.pyplot as plt

from barrels_detection import BarrelsDetection
from visualization import show_color

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Barrels Detection')
    parser.add_argument('images', metavar='IMAGE', help='image path', nargs="+")

    args = parser.parse_args()

    detection = BarrelsDetection()

    for path in args.images:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        d, detection_image = detection.detect(image)
        show_color(detection_image)

    plt.show()