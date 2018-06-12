import cv2
import numpy as np


class Detection:
    def __init__(self, start, end, center):
        self.start = start
        self.end = end
        self.center = center


class BarDetection:
    def __init__(self):
        self.bottom_cut_off = 140 #70
        self.top_cut_off = 120 #60
        self.lower_color_bound = np.array([10, 100, 0])  # hsv space
        self.upper_color_bound = np.array([50, 255, 255])  # hsv space
        self.min_contour_area = 150

    def preprocess(self, img):
        # cut off bottom and top
        img_cut = img[self.top_cut_off:-self.bottom_cut_off, :, :]

        # convert to hsv and mask out colors out of the defined range
        hsv_img = cv2.cvtColor(img_cut, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, self.lower_color_bound, self.upper_color_bound)

        filtered_img = cv2.bitwise_and(img_cut, img_cut, mask=mask)

        return img_cut, mask, filtered_img

    def postprocess(self, original_img, detection_img):
        post_img = original_img.copy()
        post_img[self.top_cut_off:-self.bottom_cut_off, :, :] = detection_img.copy()
        return post_img

    def contour_detection(self, mask, detection_image):
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = filter(lambda cnt: cv2.contourArea(cnt) > self.min_contour_area, contours)

        if len(filtered_contours) != 2:
            filtered_contours = []

        img_with_contours = cv2.drawContours(detection_image, filtered_contours, -1, (255, 0, 0), 2)

        return filtered_contours, img_with_contours

    def fit_lines(self, contours, detection_image):
        detections = []
        rows, cols = detection_image.shape[:2]
        for cnt in contours:
            [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((cols - x) * vy / vx) + y)
            detection_image = cv2.line(detection_image, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

            robot_center = np.array([rows, cols / 2.0])
            rect = cv2.minAreaRect(cnt)
            max_dimension = max(rect[1])
            normalized_dir = np.array([vx[0], vy[0]])
            base_vec = np.array([x[0], y[0]])
            plus_vec = base_vec + max_dimension / 2.0 * normalized_dir
            minus_vec = base_vec - max_dimension / 2.0 * normalized_dir

            plus_norm = np.linalg.norm(plus_vec - robot_center)
            minus_norm = np.linalg.norm(minus_vec - robot_center)

            start_vec = plus_vec if plus_norm < minus_norm else minus_vec
            end_vec = minus_vec if plus_norm < minus_norm else plus_vec

            detection_image = cv2.drawMarker(detection_image, (start_vec[0], start_vec[1]), (0, 255, 255),
                                             markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=1,
                                             line_type=cv2.LINE_AA)
            detection_image = cv2.drawMarker(detection_image, (end_vec[0], end_vec[1]), (0, 255, 255),
                                             markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=1,
                                             line_type=cv2.LINE_AA)

            start_global_vec = start_vec + np.array([0, self.top_cut_off])
            end_global_vec = end_vec + np.array([0, self.top_cut_off])
            center_global = base_vec + np.array([0, self.top_cut_off])
            d = Detection(start_global_vec, end_global_vec, center_global)
            detections.append(d)
        return detection_image, detections

    def detect(self, img):
        detection_image, mask, _ = self.preprocess(img)
        contours, detection_image = self.contour_detection(mask, detection_image)
        detection_image, detections = self.fit_lines(contours, detection_image)
        post_img = self.postprocess(img, detection_image)
        return post_img, detections
