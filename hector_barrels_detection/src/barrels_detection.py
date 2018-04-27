import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

from visualization import show_color, show_gray, gray_to_color


class Detection:
    def __init__(self, name, center, points):
        self.name = name
        self.center = center
        self.points = points


class BarrelsDetection:
    def __init__(self):
        self.bottom_cut_off = 150
        self.threshold = 0.05
        self.dilate_ksize = 5

    def detect(self, img):
        # cut off bottom
        img_cut = img[:img.shape[0] - self.bottom_cut_off, :, :]
        # to float
        img_float = img_cut.astype(float) / 255.0
        # show_color(img_float)

        # only blue channel
        img_red = img_float[:, :, 0]
        img_green = img_float[:, :, 1]
        img_blue = img_float[:, :, 2]
        img_only_blue = np.clip(img_blue - img_green - img_red, 0, 1)
        # show_gray(img_only_blue)

        # thresholding
        th, img_thresh = cv2.threshold(img_only_blue, self.threshold, 1, cv2.THRESH_BINARY)
        # show_gray(img_thresh)
        # to uchar
        img_thresh = img_thresh.astype(np.uint8) * 255

        # dilate
        kernel = np.ones((self.dilate_ksize, self.dilate_ksize), np.uint8)
        img_dilated = cv2.dilate(img_thresh, kernel, iterations=2)
        # show_gray(img_dilated)

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 255
        params.thresholdStep = 254

        # distance
        params.minDistBetweenBlobs = 50

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 1000
        params.maxArea = 13000

        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False

        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        img_inv = 255 - img_dilated
        keypoints = detector.detect(img_inv)
        # print "Found blobs:", len(keypoints)

        # img_blobs = gray_to_color(img_dilated)

        # img_detect = \
        #     cv2.drawKeypoints(img, keypoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # img_blobs = \
        #     cv2.drawKeypoints(img_blobs, keypoints, np.array([]), (255, 0, 0),
        #                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # show_gray(img_blobs)

        detections = list()
        img_detect = img_cut.copy()
        for k in keypoints:
            # approximate circle with rectangle
            l = k.size / 2
            c1 = k.pt[0]
            c2 = k.pt[1]
            pt1 = (int(c1 - l), int(c2 - l))
            pt2 = (int(c1 - l), int(c2 + l))
            pt3 = (int(c1 + l), int(c2 + l))
            pt4 = (int(c1 + l), int(c2 - l))

            d = Detection("barrel", [c1, c2], [pt1, pt2, pt3, pt4])
            detections.append(d)

            cv2.rectangle(img_detect, pt1, pt3, (255, 0, 0), thickness=2)

        show_color(img_detect)

        return detections, img_detect
