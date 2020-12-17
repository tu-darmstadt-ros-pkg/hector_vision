import cv2
import numpy as np
import math

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
        self.min_area = 1500

    def find_white_barrels(self, img):
        # cut off bottom
        img_cut = img[:img.shape[0] - self.bottom_cut_off, :, :]

        # downscale
        img_lr = cv2.pyrDown(img)
        img_lr = cv2.pyrDown(img_lr)
        img_lr = cv2.pyrDown(img_lr)

        # LAB
        img_lab = cv2.cvtColor(img_lr, cv2.COLOR_RGB2Lab)

        lower = (150, 0, 0)
        upper = (256, 131, 130)
        mask = cv2.inRange(img_lab, lower, upper)
        img_lab_masked = cv2.bitwise_and(img_lab, img_lab, mask=mask)
        img_lab_gray = cv2.cvtColor(img_lab_masked, cv2.COLOR_LAB2RGB)
        th, img_lab_thresh = cv2.threshold(img_lab_gray[:, :, 0], 100, 255, cv2.THRESH_BINARY)

        # dilate
        ksize = 3
        kernel = np.ones((ksize, ksize), np.uint8)
        img_dilated = cv2.dilate(img_lab_thresh, kernel, iterations=1)
        return img_dilated, img_cut, 8

    def find_blue_barrels(self, img):
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
        img_dilated = cv2.dilate(img_thresh, kernel, iterations=3)
        # show_gray(img_dilated)
        #img_closing = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

        return img_dilated, img_cut, 1

    def contour_detection(self, img, detection_image, scaling):
        # Find contours in binary image
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area
        contours_filtered = []
        for c in contours:
            if cv2.contourArea(c) > self.min_area:
                for p in c:
                    p[0] *= scaling
                contours_filtered.append(c)
        # print "Filtered contour count", len(contours_filtered)
        cv2.drawContours(detection_image, contours_filtered, -1, (0, 255, 0), 3)

        # Approximate contours
        approx_contours = []
        for contour in contours_filtered:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            approx_contours.append(approx)

        # Find contour centers
        centers = []
        for c in contours_filtered:
            sum = [0, 0]
            for p in c:
                sum[0] += p[0][0]
                sum[1] += p[0][1]
            point = (sum[0] // len(c), sum[1] // len(c))
            cv2.circle(detection_image, point, 4, (255, 255, 255), thickness=4)
            centers.append(point)

        detections = []
        for contour, center in zip(approx_contours, centers):
            points = [[p[0][0], p[0][1]] for p in contour]

            # approximate circle with rectangle
            d = Detection("barrel", center, points)
            detections.append(d)

        return detections, detection_image

    def blob_detection(self, img, detection_image):
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
        params.minArea = self.min_area
        params.maxArea = 13000

        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False

        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        img_inv = 255 - img
        keypoints = detector.detect(img_inv)

        detection_image = \
            cv2.drawKeypoints(img, keypoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        detections = list()
        for k in keypoints:
            # approximate circle with rectangle
            l = k.size // 2
            c1 = k.pt[0]
            c2 = k.pt[1]
            pt1 = (int(c1 - l), int(c2 - l))
            pt2 = (int(c1 - l), int(c2 + l))
            pt3 = (int(c1 + l), int(c2 + l))
            pt4 = (int(c1 + l), int(c2 - l))

            d = Detection("barrel", [c1, c2], [pt1, pt2, pt3, pt4])
            detections.append(d)

            cv2.rectangle(detection_image, pt1, pt3, (255, 0, 0), thickness=2)

        return detections, detection_image

    def detect(self, img, barrel_type):
        if barrel_type == "blue":
            img_pre, detection_image, scaling = self.find_blue_barrels(img)
        elif barrel_type == "white":
            img_pre, detection_image, scaling = self.find_white_barrels(img)
        else:
            print("Unknown barrel type '{}'".format(barrel_type))
            return [], img
        detections, detection_image = self.contour_detection(img_pre, detection_image, scaling)
        return detections, detection_image
