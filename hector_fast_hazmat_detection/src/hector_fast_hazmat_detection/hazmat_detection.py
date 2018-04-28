from __future__ import print_function, division
import cv2
import numpy as np
import os

from timeit import default_timer as timer

class DebugInformation:
    best_matches = None  # type: List[(np.ndarray, np.ndarray, np.ndarray, float)]
    detections = None  # type: List[(np.ndarray, np.ndarray, np.ndarray, float)]
    sub_images = None  # type: List[(np.ndarray, np.ndarray, np.ndarray)]
    filtered_contours = None  # type: List[np.ndarray]
    approx_contours = None  # type: List[np.ndarray]
    contours = None  # type: List[np.ndarray]
    dilated_image = None  # type: np.ndarray
    edge_image = None  # type: np.ndarray

    def __init__(self):
        self.edge_image = None
        self.dilated_image = None
        self.contours = []
        self.approx_contours = []
        self.filtered_contours = []
        self.sub_images = []
        self.detections = []
        self.best_matches = []
        self.rejections = []
        self.matches = []


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


def get_distance_squared(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2


def edgedetect (channel):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)

    sobel[sobel > 255] = 255  # Some values seem to go above 255. However RGB channels has to be within 0-255
    return sobel


#def is_color(image):
#    image_gray = np.sum(image, axis=2) / 3
#    diff = image - image_gray[:, :, np.newaxis]
#    return ((len(diff[diff < 25])) / np.prod(image.shape)) < 0.75
def is_color(image):
    image = image.astype(np.int)
    rg = (image[:, :, 0] - image[:, :, 1])**2
    rb = (image[:, :, 0] - image[:, :, 2])**2
    gb = (image[:, :, 1] - image[:, :, 2])**2
    val = np.sum(rg + rb + gb) / (np.prod(image.shape) * 255)
    if 0.4 < val <= 4:
        return None
    return val > 2


def calculate_color_correlation(img, template):
    return cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)[0, 0]
    img = img.astype(np.int)
    template = template.astype(np.int)
    norm1 = img / np.sqrt(np.sum(img**2))
    norm2 = template / np.sqrt(np.sum(template**2))
    return np.sum(norm1 * norm2)


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

    def get_edge_image(self, image, image_mem=None):
        """
        Retrieves the edge image for the given input image
        :param image: The image which may be a numpy array or a cv2.UMat
        :param image_mem: The image array which has to be a numpy array. If it is None, image is used and has to be a
         numpy array
        :return: Either a numpy array or a cv2.UMat containing the binary edge image for the given input image
        """
        if image_mem is None:
            image_mem = image
        resized = image
        for i in range(self.downsample_passes+1):
            resized = cv2.pyrDown(resized)
        resized = cv2.pyrUp(resized)
        #resized = cv2.GaussianBlur(image, (5, 5), 0)
        #resized = cv2.resize(resized,
        #                     (image_mem.shape[1] // self.resolution_divider,
        #                      image_mem.shape[0] // self.resolution_divider))
        return cv2.Canny(resized, 60, 40)

    @staticmethod
    def morph_image(image):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        morph = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
        morph = cv2.dilate(cv2.erode(morph, None), None)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=2)
        return morph

#    def filter_edge_image(self, edge_image):
 #       _, contours, _ = cv2.findContours(edge_image, cv2.RET)

    def find_and_filter_regions_of_interest(self, morphed_image, image_mem, debug_info):
        _, contours, _ = cv2.findContours(morphed_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if debug_info is not None:
            debug_info.contours = contours
        regions_of_interest = []
        margin = 15

        # merge close contours
        # bounding_boxes = []
        # for c in contours:
        #     bounding_boxes.append(cv2.boundingRect(c))
        #
        # i = len(bounding_boxes) - 1
        # while i >= 0:
        #     for j in range(len(bounding_boxes)-1, -1, -1):
        #         if i == j:
        #             continue
        #         (x1, y1, w1, h1) = bounding_boxes[i]
        #         (x2, y2, w2, h2) = bounding_boxes[j]
        #         c1_x = x1 + w1 // 2
        #         c2_x = x2 + w2 // 2
        #         c1_y = y1 + h1 // 2
        #         c2_y = y2 + h2 // 2
        #         if (c1_x - c2_x)**2 + (c1_y - c2_y)**2 < 400:
        #             merge = False
        #             for pt1 in contours[i]:
        #                 for pt2 in contours[j]:
        #                     if (pt1[0, 0] - pt2[0, 0])**2 + (pt1[0, 1] - pt2[0, 1])**2 < 100:
        #                         merge = True
        #             if merge:
        #                 contours[i] = np.vstack((contours[i], contours[j]))
        #                 bounding_boxes[i] = cv2.boundingRect(contours[i])
        #                 del contours[j]
        #                 del bounding_boxes[j]
        #                 if j < i:
        #                     i -= 1
        #     i -= 1


        # sort contours by size descending first
        sorted_contours = []
        sizes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            inserted = False
            for i in range(len(sizes)):
                if sizes[i] < area:
                    sorted_contours.insert(i, contour)
                    sizes.insert(i, area)
                    inserted = True
                    break
            if not inserted:
                sizes.append(area)
                sorted_contours.append(contour)

        for contour in sorted_contours:
            contour *= (2**self.downsample_passes)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            if len(approx) < 4:
                continue
            approx = cv2.convexHull(approx)

            reject = False
            for pt in approx:
                pt = pt[0]
                if pt[0] < 5 or pt[0] > image_mem.shape[1] - 5 or pt[1] < 5 or pt[1] > image_mem.shape[0] - 5:
                    reject = True
                    break
            if reject:
                continue

            (x, y, w, h) = cv2.boundingRect(approx)
            for roi in regions_of_interest:
                if roi[0] <= x and x+w <= roi[0] + roi[2] and roi[1] < y and y+h <= roi[1] + roi[3]:
                    reject = True
                    break
            if reject:
                continue
            ar = w / float(h)
            if ar < 0.6 or ar > 1.1:
                continue
            area = cv2.contourArea(approx)
            if len(approx) > 5 and area > 4000:
                continue
            if abs(area - w * h) / np.maximum(area, w * h) > 0.75:
                continue
            if area < 800 or area > 180 ** 2: continue
            if debug_info is not None:
                debug_info.filtered_contours.append(approx)
            x_offset = x - margin if x >= margin else 0
            y_offset = y - margin if y >= margin else 0
            w = w + 2 * margin if x_offset + w + 2 * margin <= image_mem.shape[1] else image_mem.shape[1] - x_offset
            h = h + 2 * margin if y_offset + h + 2 * margin <= image_mem.shape[0] else image_mem.shape[0] - y_offset

            if debug_info is not None:
                debug_info.approx_contours.append(approx)
            regions_of_interest.append((x_offset, y_offset, w, h))
        return regions_of_interest

    def get_rectangles(self, regions_of_interest, image_mem, debug_info):
        filtered_sub_images = []
        rectangles = []
        sub_contours = []
        for roi in regions_of_interest:
            (x_offset, y_offset, w, h) = roi
            sub_image = image_mem[y_offset:y_offset + h, x_offset:x_offset + w, :]
            test_blur = cv2.GaussianBlur(sub_image, (5, 5), 0)
            test_edges = np.max(np.array([edgedetect(test_blur[:, :, 0]),
                                          edgedetect(test_blur[:, :, 1]),
                                          edgedetect(test_blur[:, :, 2])]), axis=0)
            mean = np.mean(test_edges)
            test_edges[test_edges <= mean] = 0
            _, test_edges = cv2.threshold(test_edges, 80, 255, cv2.THRESH_BINARY)
            test_edges = test_edges.astype(np.uint8)

            # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            #test_edges = cv2.morphologyEx(test_edges, cv2.MORPH_CLOSE, kernel, iterations=3).astype(np.uint8)
            #test_edges = cv2.erode(test_edges, None)
            _, contours, _ = cv2.findContours(test_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            if len(filtered) == 0:
                if debug_info is not None:
                    debug_info.sub_images.append((sub_image, test_edges, None))
                continue
            contours = np.vstack(c for c in filtered) if len(filtered) > 1 else filtered[0]
            approx = cv2.convexHull(contours)
            peri = cv2.arcLength(approx, True)
            approx = cv2.approxPolyDP(approx, 0.04 * peri, True)

            if debug_info is not None:
                debug_info.sub_images.append((sub_image, test_edges, approx))
            if len(approx) != 4:
                continue

            max_distance_squared = get_distance_squared(approx[-1, 0, :], approx[0, 0, :])
            for i in range(len(approx) - 1):
                distance_squared = get_distance_squared(approx[i, 0, :], approx[i + 1, 0, :])
                if distance_squared > max_distance_squared:
                    max_distance_squared = distance_squared

            size = int(np.sqrt(max_distance_squared))
            if size < 24:
                continue

            left_most = approx[0, 0, :]
            top_most = approx[0, 0, :]
            right_most = approx[0, 0, :]
            for pt in approx[:, 0, :]:
                if pt[0] < left_most[0]:
                    left_most = pt
                elif pt[1] < top_most[1]:
                    top_most = pt
                elif pt[0] > right_most[0]:
                    right_most = pt
            pts1 = np.float32([left_most, top_most, right_most])
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
        if matches > 4:
            return correlation + 0.25 + matches * 0.01
        return correlation + matches * 0.01

    def detect(self, image, debug=False):  # type: (np.ndarray, bool) -> DetectionResult
        result = DetectionResult()
        image = image[0:450, :, :]
        image_mem = image
        if self.gpu:
            image = cv2.UMat(image)
        edge_image = self.get_edge_image(image, image_mem)
        #morph = self.morph_image(edge_image)
        morph = edge_image
        if debug:
            result.debug_information = DebugInformation()
            result.debug_information.edge_image = edge_image
            result.debug_information.dilated_image = morph
        regions_of_interest = self.find_and_filter_regions_of_interest(morph, image_mem,
                                                                       result.debug_information)
        sub_images, contours, rectangles = self.get_rectangles(regions_of_interest, image_mem,
                                                               result.debug_information)

        for i in range(len(rectangles)):
            rectangle = rectangles[i]
            rectangle_is_color = is_color(rectangle)
            corr_soft_threshold = 0.3
            corr_threshold = 0.7 if rectangle_is_color is None or rectangle_is_color else 0.7
            match_soft_threshold = 1
            min_sift_matches = 6 if rectangle_is_color is not None or rectangle_is_color else 8

            start = timer()
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
                if (rectangle_is_color and sign.is_color) is not None and sign.is_color != rectangle_is_color:
                    continue
                template = cv2.resize(sign.image, (rectangle.shape[1], rectangle.shape[0]))
                matches = get_sift_matches(target_keypoints, target_descriptors, sign)

                template_blur = cv2.GaussianBlur(template, (5, 5), 0)
                correlation = calculate_color_correlation(rectangle, template_blur)
                ew = self.effective_weight(correlation, matches)
                best_sign_corr = correlation
                if ew > second_best_effective_weight:
                    if ew > best_effective_weight:
                        second_best_effective_weight = best_effective_weight
                        best_correlation = correlation
                        best_matches = matches
                        best_effective_weight = ew
                        best_sign = sign
                        best_template = template
                    else:
                        second_best_effective_weight = ew
                # for rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                #     rotated = cv2.rotate(template_blur, rotation)
                #     correlation = calculate_color_correlation(rectangle, rotated)
                #     ew = self.effective_weight(correlation, matches)
                #     if correlation > best_sign_corr:
                #         best_sign_corr = correlation
                #     if ew > second_best_effective_weight:
                #         if ew > best_effective_weight:
                #             second_best_effective_weight = best_effective_weight
                #             best_correlation = correlation
                #             best_matches = matches
                #             best_effective_weight = ew
                #             best_sign = sign
                #             best_template = rotated
                #         else:
                #             second_best_effective_weight = ew
                if debug:
                    sign_values.append((template, best_sign_corr, matches))
            if debug:
                result.debug_information.matches.append((sub_images[i], rectangle, sign_values))
                result.debug_information.best_matches.append((sub_images[i], rectangle, best_template,
                                                              (best_correlation, best_matches, rectangle_is_color)))
            if (0.7 * best_effective_weight <= second_best_effective_weight or best_correlation < corr_soft_threshold
                or best_matches < match_soft_threshold) and\
                    best_correlation < corr_threshold and best_matches < min_sift_matches:
                continue
            if debug:
                result.debug_information.detections.append((sub_images[i], rectangle, best_template,
                                                            (best_correlation, best_matches, rectangle_is_color)))
            result.detections.append(Detection(best_sign, contours[i]))
        return result


    # def detect(self, image, debug=False):  # type: (np.ndarray, bool) -> DetectionResult
    #     image_mem = image
    #     image = cv2.UMat(image_mem) if self.gpu else image_mem
    #     result = DetectionResult()
    #
    #     resized = cv2.resize(cv2.GaussianBlur(image, (5, 5), 0), (image_mem.shape[1] // 2, image_mem.shape[0] // 2))
    #     edges = cv2.Canny(resized, 80, 30)
    #     if debug:
    #         result.debug_information = DebugInformation()
    #         result.debug_information.edge_image = edges
    #
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))  # for half resolution
    #     morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    #     morph = cv2.dilate(cv2.erode(morph, None), None)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))  # for half resolution
    #     morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=2)
    #     if debug:
    #         result.debug_information.dilated_image = morph
    #
    #     _, contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     for i in range(len(contours)):
    #         contours[i] = contours[i] * 2
    #     if debug:
    #         result.debug_information.contours = contours
    #
    #     for contour in contours:
    #         # Filter contours
    #         peri = cv2.arcLength(contour, True)
    #         approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    #         if debug:
    #             result.debug_information.approx_contours.append(approx)
    #         # Remove contours close to image boundary
    #         reject = False
    #         for pt in approx:
    #             pt = pt[0]
    #             if pt[0] < 5 or pt[0] > image_mem.shape[1] - 5 or pt[1] < 5 or pt[1] > image_mem.shape[0] - 5:
    #                 reject = True
    #                 break
    #         if reject:
    #             continue
    #
    #         (x, y, w, h) = cv2.boundingRect(approx)
    #         ar = w / float(h)
    #         if ar < 0.25 or ar > 2.5: continue
    #         area = cv2.contourArea(approx)
    #         if len(approx) < 4:
    #             continue
    #         if len(approx) > 5 and area > 4000:
    #             continue
    #         if abs(area - w * h) / np.maximum(area, w*h) > 0.75:
    #             continue
    #         if area < 200 or area > 120**2:
    #             continue
    #         if debug:
    #             result.debug_information.filtered_contours.append(approx)
    #
    #         # Get sub image
    #         margin = 40
    #         x_offset = x - margin if x >= margin else 0
    #         y_offset = y - margin if y >= margin else 0
    #         w = w + 2 * margin if x_offset + w + 2 * margin  <= image_mem.shape[1] else image_mem.shape[1] - x_offset
    #         h = h + 2 * margin if y_offset + h + 2 * margin <= image_mem.shape[0] else image_mem.shape[0] - y_offset
    #         sub_image = image_mem[y_offset:y_offset+h, x_offset:x_offset+w, :]
    #
    #         # Locate rectangle in sub image
    #         sub_blur = cv2.medianBlur(sub_image, 5)
    #         for strengthen in [True, False]:
    #             test_edges = np.max( np.array([edgedetect(sub_blur[:, :, 0], strengthen),
    #                                            edgedetect(sub_blur[:, :, 1], strengthen),
    #                                            edgedetect(sub_blur[:, :, 2], strengthen)]), axis=0)
    #             mean = np.mean(test_edges)
    #             test_edges[test_edges <= mean] = 0
    #             test_edges = test_edges.astype(np.uint8)
    #             if self.gpu:
    #                 test_edges = cv2.UMat(test_edges)
    #             _, test_edges = cv2.threshold(test_edges, 80, 255, cv2.THRESH_BINARY)
    #             kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    #             test_edges = cv2.morphologyEx(test_edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    #             _, contours, _ = cv2.findContours(test_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #             filtered = []
    #             for c in contours:
    #                 reject = False
    #                 if cv2.contourArea(c) < 32:
    #                     continue
    #                 for pt in c:
    #                     pt = pt[0, :]
    #                     if pt[0] < 4 or pt[0] > sub_image.shape[1] - 4 or pt[1] < 4 or pt[1] > sub_image.shape[0] - 4:
    #                         reject = True
    #                         break
    #                 if reject:
    #                     continue
    #                 filtered.append(c)
    #             if len(filtered) == 0 or len(filtered) > 4:
    #                 if debug:
    #                     result.debug_information.sub_images.append((sub_image, test_edges, None))
    #                 continue
    #             contours = np.vstack(c for c in filtered) if len(filtered) > 1 else filtered[0]
    #             approx = cv2.convexHull(contours)
    #             peri = cv2.arcLength(approx, True)
    #             approx = cv2.approxPolyDP(approx, 0.04 * peri, True)
    #             if debug:
    #                 result.debug_information.sub_images.append((sub_image, test_edges, approx))
    #
    #             if len(approx) != 4:
    #                 continue
    #
    #             max_distance_squared = get_distance_squared(approx[-1, 0, :], approx[0, 0, :])
    #             for i in range(len(approx) - 1):
    #                 distance_squared = get_distance_squared(approx[i, 0, :], approx[i + 1, 0, :])
    #                 if distance_squared > max_distance_squared:
    #                     max_distance_squared = distance_squared
    #
    #             size = int(np.sqrt(max_distance_squared))
    #             if size < 32:
    #                 continue
    #
    #             left_most = approx[0, 0, :]
    #             top_most = approx[0, 0, :]
    #             right_most = approx[0, 0, :]
    #             for pt in approx[:, 0, :]:
    #                 if pt[0] < left_most[0]:
    #                     left_most = pt
    #                 elif pt[1] < top_most[1]:
    #                     top_most = pt
    #                 elif pt[0] > right_most[0]:
    #                     right_most = pt
    #             pts1 = np.float32([left_most, top_most, right_most])
    #             pts2 = np.float32([[-4, -4], [size+3, -4], [size+3, size+3]])
    #             M = cv2.getAffineTransform(pts1, pts2)
    #             target = cv2.warpAffine(sub_image, M, (size, size))
    #             #target = cv2.GaussianBlur(target, (5, 5), 0)
    #             target_kp, target_dsc = self.sift.detectAndCompute(target, None)
    #             best_difference = 0
    #             best_template = None
    #             best_metrics = None
    #             best_sign = None
    #             for sign in self.signs:
    #                 if sign.name != "corrosive":
    #                     continue
    #                 test = cv2.resize(sign.image, (size, size))
    #                 test = cv2.GaussianBlur(test, (5, 5), 0)
    #                 diff = calculate_difference(target, test)
    #                 test_kp, test_dsc = self.sift.detectAndCompute(test, None)
    #                 matches = self.matcher.knnMatch(target_dsc, test_dsc, 2)
    #                 good = 0
    #                 for m, n in matches:
    #                     if m.distance < 0.8 * n.distance:
    #                         good += 1
    #                 combined_diff = diff + good / 40
    #                 if diff > best_difference:
    #                     best_difference = combined_diff
    #                     if debug:
    #                         best_template = test
    #                         best_metrics = (diff, good)
    #                     best_sign = sign
    #                 for rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
    #                     rotated = cv2.rotate(test, rotation)
    #                     diff = calculate_difference(target, rotated)
    #                     combined_diff = diff + good / 40
    #                     if diff > best_difference:
    #                         best_difference = combined_diff
    #                         if debug:
    #                             best_template = rotated
    #                             best_metrics = (diff, good)
    #                         best_sign = sign
    #                 if sign.name == "corrosive":
    #                     break
    #             if debug:
    #                 result.debug_information.best_matches.append((sub_image, target, best_template, best_metrics))
    #             if best_difference < 1:
    #                 continue
    #             if debug:
    #                 result.debug_information.detections.append((sub_image, target, best_template, best_metrics))
    #             result.detections.append(Detection(best_sign.name, approx + np.array([x_offset, y_offset])))
    #             break
    #     return result
