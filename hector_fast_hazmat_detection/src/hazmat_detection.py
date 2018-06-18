from __future__ import print_function, division
import cv2
import numpy as np
import os
import hector_vision

from timeit import default_timer as timer
from area_of_interest_detection import detect_areas_of_interest
from debug_information import DebugInformation
from distance_measures import *
from edge_detection import *


def is_qr_code(edge_image):
    ratio_white = np.sum(edge_image) / (np.prod(edge_image) * 255)
    return ratio_white > 0.6


def is_color(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    mean, stddev = cv2.meanStdDev(lab_image)
    sum_stddev = np.sum(stddev[1:])
    sum_mean = np.sum((mean[1:] - 128) ** 2)
    if sum_stddev < 8 and sum_mean < 200:
        return False
    if sum_stddev < 30 and sum_mean < 600:
        return None
    return True


def calculate_color_correlation(img, template):
    return cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)


def get_sift_matches(kp, dsc, sign):  # type: (np.ndarray, np.ndarray, HazmatSign) -> int
    if dsc is None or len(dsc) < 2 or sign.descriptors is None or len(sign.descriptors) < 2:
        return 0
    matches = sign.matcher.knnMatch(dsc, sign.descriptors, 2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) < 4:
        return len(good)
    else:
        src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([sign.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            return 4
        return len(good)


class DetectionSettings:
    def __init__(self):
        self.canny_higher_threshold = 40
        self.canny_lower_threshold = 20
        self.dilate_iterations = 4
        self.erode_iterations = 4


class Detection:
    name = None  # type: str
    sign = None # type: HazmatSign
    contour = None  # type: np.ndarray

    def __init__(self, sign, contour):
        self.name = sign.name
        self.sign = sign
        self.contour = contour


class DetectionResult:
    debug_information = None  # type: DebugInformation
    detections = None  # type: List[Detection]

    def __init__(self):
        self.detections = []
        self.debug_information = None


class HazmatSign:
    def __init__(self, name, image, keypoints, descriptors):
        self.name = name
        self.image = image
        self.keypoints = keypoints
        self.descriptors = descriptors
        index_params = dict(algorithm=0, trees=4)
        search_params = dict(checks=32)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.matcher.add(descriptors)
        self.matcher.train()
        self.is_color = is_color(image)


class HazmatSignDetector:
    def __init__(self, hazmat_sign_folder, gpu=False):
        self.gpu = gpu
        self.downsample_passes = 2  # Each downsample pass halfes the resolution
        self.signs = []
        self.sift = cv2.xfeatures2d.SIFT_create()
        for f in os.listdir(hazmat_sign_folder):
            if not os.path.isfile(os.path.join(hazmat_sign_folder, f)):
                continue
            img = cv2.imread(os.path.join(hazmat_sign_folder, f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            kp, dsc = self.sift.detectAndCompute(img, None)
            self.signs.append(HazmatSign(os.path.splitext(f)[0], img, kp, dsc))

    def get_contour(self, rectangle_edges, w, h, sub_image):
        _, contours, _ = cv2.findContours(rectangle_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = []
        for c in contours:
            reject = False
            c = cv2.convexHull(c)
            if cv2.contourArea(c) < w * h * 0.01:
                continue
            for pt in c:
                pt = pt[0, :]
                if pt[0] < 4 or pt[0] > sub_image.shape[1] - 4 or pt[1] < 4 or pt[1] > sub_image.shape[0] - 4:
                    reject = True
                    break
            if reject:
                continue
            filtered.append(c)
        contours = np.vstack(c for c in filtered) if len(filtered) > 1 else filtered[0]
        approx = cv2.convexHull(contours)
        peri = cv2.arcLength(approx, True)
        approx = cv2.approxPolyDP(approx, 0.04 * peri, True)

    def get_rectangles(self, regions_of_interest, image_mem, debug_info):
        filtered_sub_images = []
        rectangles = []
        sub_contours = []
        for roi in regions_of_interest:
            (x_offset, y_offset, w, h) = roi
            sub_image = image_mem[y_offset:y_offset + h, x_offset:x_offset + w, :]
            test_edges = hector_vision.color_difference_map(sub_image)
            upper, lower = hector_vision.calculateThresholds(test_edges)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
            test_edges = cv2.erode(hector_vision.threshold(test_edges, upper, lower), kernel)

            _, contours, _ = cv2.findContours(test_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_area = 0
            largest = None
            for c in contours:
                hull = cv2.convexHull(c)
                area = cv2.contourArea(hull)
                if area > largest_area:
                    largest = hull
                    largest_area = area
            if largest is None:
                if debug_info is not None:
                    debug_info.sub_images.append((sub_image, test_edges, None))
                continue
            approx = largest
            peri = cv2.arcLength(approx, True)
            approx = cv2.approxPolyDP(approx, 0.06 * peri, True)

            if debug_info is not None:
                debug_info.sub_images.append((sub_image, test_edges, approx))
            if len(approx) != 4:
                continue

            max_distance_squared = squared_distance(approx[-1, 0, :], approx[0, 0, :])
            for i in range(len(approx) - 1):
                distance_squared = squared_distance(approx[i, 0, :], approx[i + 1, 0, :])
                if distance_squared > max_distance_squared:
                    max_distance_squared = distance_squared

            size = int(np.sqrt(max_distance_squared))
            if size < 24:
                continue

            x_sorted_points = []
            for pt in approx[:, 0, :]:
                inserted = False
                for i in range(len(x_sorted_points)):
                    if pt[0] < x_sorted_points[i][0]:
                        x_sorted_points.insert(i, pt)
                        inserted = True
                        break
                if not inserted:
                    x_sorted_points.append(pt)

            # Determine orientation (45 or 0 + n * 90 degrees)
            if 2 * (x_sorted_points[1][0] - x_sorted_points[0][0]) > x_sorted_points[2][0] - x_sorted_points[1][0]:
                # If it is approximately 45 degrees rotated, the leftmost is the bottom left, the topmost is the top
                #  left and the rightmost is the top right
                bottom_left = x_sorted_points[0]
                top_left = x_sorted_points[1] if x_sorted_points[1][1] < x_sorted_points[2][1] else x_sorted_points[2]
                top_right = x_sorted_points[3]
            else:
                # Otherwise we determine which one is top or bottom by comparing the y values of the two leftmost and
                #  two rightmost points
                top_left, bottom_left = (x_sorted_points[0], x_sorted_points[1])\
                    if x_sorted_points[0][1] < x_sorted_points[1][1] else (x_sorted_points[1], x_sorted_points[0])
                top_right, bottom_right = (x_sorted_points[2], x_sorted_points[3])\
                    if x_sorted_points[2][1] < x_sorted_points[3][1] else (x_sorted_points[3], x_sorted_points[2])
            pts1 = np.float32([bottom_left, top_left, top_right])
            pts2 = np.float32([[-2, size + 1], [-2, -2], [size + 1, -2]])
            M = cv2.getAffineTransform(pts1, pts2)
            out_image = sub_image.copy()
            out_image = cv2.warpAffine(out_image, M, (size, size))
            approx = approx + np.array([x_offset, y_offset])
            sub_contours.append(approx.astype(np.int))
            rectangles.append(out_image)
            filtered_sub_images.append(sub_image)
        return filtered_sub_images, sub_contours, rectangles

    @staticmethod
    def effective_weight(correlation, matches):
        correlation = (correlation + 1) / 2
        if matches > 4:
            return correlation + 0.25 + matches * 0.02
        return correlation + matches * 0.02

    def detect(self, image, debug=False):  # type: (np.ndarray, bool) -> DetectionResult
        result = DetectionResult()
        image = image[0:450, :, :]
        image_mem = image
        if self.gpu:
            image = cv2.UMat(image)
        if debug:
            result.debug_information = DebugInformation()
        contours, regions_of_interest = detect_areas_of_interest(image, self.downsample_passes, result.debug_information)
        sub_images, contours, rectangles = self.get_rectangles(regions_of_interest, image_mem,
                                                               result.debug_information)

        for i in range(len(rectangles)):
            rectangle = rectangles[i]
            rectangle_is_color = is_color(rectangle)
            corr_soft_threshold = 0.3
            corr_threshold = 0.65 if rectangle_is_color is None or rectangle_is_color else 0.75
            match_soft_threshold = 1
            min_sift_matches = 5 if rectangle_is_color is not None or rectangle_is_color else 8

            target_keypoints, target_descriptors = self.sift.detectAndCompute(sub_images[i], None)
            rectangle = cv2.GaussianBlur(rectangle, (5, 5), 0)

            best_correlation = -1
            best_matches = -1
            best_effective_weight = -1
            best_sign = None
            best_template = None
            second_best_effective_weight = -1
            if debug:
                sign_values = []
            for sign in self.signs:
                if rectangle_is_color is not None and sign.is_color is not None and sign.is_color != rectangle_is_color:
                    continue
                template = cv2.resize(sign.image, (rectangle.shape[1], rectangle.shape[0]))
                matches = get_sift_matches(target_keypoints, target_descriptors, sign)

                template_blur = cv2.GaussianBlur(template, (5, 5), 0)
                correlation = calculate_color_correlation(rectangle, template_blur)
                best_sign_corr = correlation
                for rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                    rotated = cv2.rotate(rectangle, rotation)
                    correlation = calculate_color_correlation(rotated, template_blur)
                    if correlation > best_correlation:
                        best_correlation = correlation
                ew = self.effective_weight(best_sign_corr, matches)
                if ew > second_best_effective_weight:
                    if ew > best_effective_weight:
                        second_best_effective_weight = best_effective_weight
                        best_correlation = best_sign_corr
                        best_matches = matches
                        best_effective_weight = ew
                        best_sign = sign
                        best_template = template
                    else:
                        second_best_effective_weight = ew
                if debug:
                    sign_values.append((template, best_sign_corr, matches))
            if debug:
                result.debug_information.matches.append((sub_images[i], rectangle, sign_values))
                result.debug_information.best_matches.append((sub_images[i], rectangle, best_template,
                                                              (best_correlation, best_matches, rectangle_is_color)))
            if (0.8 * best_effective_weight <= second_best_effective_weight or best_correlation < corr_soft_threshold
                or best_matches < match_soft_threshold) and\
                    best_correlation < corr_threshold and best_matches < min_sift_matches:
                continue
            if debug:
                result.debug_information.detections.append((sub_images[i], rectangle, best_template,
                                                            (best_correlation, best_matches, rectangle_is_color)))
            result.detections.append(Detection(best_sign, contours[i]))
        return result
