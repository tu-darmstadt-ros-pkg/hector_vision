#!/usr/bin/env python
import argparse
import cv2
import matplotlib.pyplot as plt

from barrels_detection import BarrelsDetection

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Barrels Detection')
    parser.add_argument('images', metavar='IMAGE', help='image path', nargs="+")

    args = parser.parse_args()

    detection = BarrelsDetection()

    for path in args.images:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        detection.detect(image)

    plt.show()