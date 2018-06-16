import cv2
import numpy as np


class DebugInfo:
    image = None
    edge_image = None
    contours = None
    filtered_contours = None
    largest_group = None
    sub_image = None
    sub_edge_image = None
    sub_contours = None

    def __init__(self):
        pass

    def get_contours_image(self):
        out_img = self.image.copy()
        cv2.drawContours(out_img, self.contours, -1, (0, 255, 0), 2)
        return out_img

    def get_filtered_contours_image(self):
        out_img = self.image.copy()
        cv2.drawContours(out_img, self.filtered_contours, -1, (255, 0, 0), 2)
        return out_img

    def get_sub_contours_image(self):
        out_img = self.sub_image.copy()
        cv2.drawContours(out_img, self.sub_contours, -1, (255, 0, 0), 2)
        return out_img


def cdm(image, threshold=None, auto_threshold_multiplier=0.5):
    image_mem = image
    mean_filtered = cv2.filter2D(image, cv2.CV_16S, np.ones((5, 5)) / 25)
    filters = np.array([[[1, 0, 0],
                         [0, 0, 0],
                         [0, 0, -1]],
                        [[0, 1, 0],
                         [0, 0, 0],
                         [0, -1, 0]],
                        [[0, 0, 1],
                         [0, 0, 0],
                         [-1, 0, 0]],
                        [[0, 0, 0],
                         [1, 0, -1],
                         [0, 0, 0]]])
    max_response = np.zeros((image_mem.shape[0], image_mem.shape[1]))
    for i in range(len(filters)):
        response = cv2.filter2D(mean_filtered, cv2.CV_16S, filters[i, :, :])
        response = np.max(np.abs(response.get() if isinstance(response, cv2.UMat) else response), axis=2)
        max_response = np.maximum(response, max_response)
    if threshold is None:
        mean = np.mean(max_response[max_response > 0])
        std = np.sqrt(np.var(max_response[max_response > 0]))
        threshold = mean + auto_threshold_multiplier * std
    _, max_response = cv2.threshold(max_response, threshold, 255, cv2.CV_8U)
    return max_response.astype(np.uint8)


def __find_circles(contours, downsample_passes, debug_info=None):
    filtered_contours = []
    areas = []
    centers = []
    for c in contours:
        c *= 2**downsample_passes
        approx = cv2.approxPolyDP(c, 0.005 * cv2.arcLength(c, True), True)
        area = cv2.contourArea(approx)
        if area < 100 or abs(cv2.arcLength(c, True) ** 2 / (4 * np.pi * cv2.contourArea(c)) - 1) > 0.2:
            continue
        moments = cv2.moments(approx)
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        inserted = False
        for i in range(len(centers)):
            if centers[i][0] > cx:
                centers.insert(i, (cx, cy))
                inserted = True
                break
        if not inserted:
            centers.append((cx, cy))
        filtered_contours.append(approx)
        areas.append(area)

    if debug_info is not None:
        debug_info.filtered_contours = filtered_contours

    if len(filtered_contours) == 0:
        return None, None

    groups = [[0]]
    for i in range(1, len(centers)):
        if np.sqrt((centers[i][0] - centers[i - 1][0]) ** 2 + (centers[i][1] - centers[i - 1][1]) ** 2) < 20:
            groups[-1].append(i)
            continue
        groups.append([i])

    # Just use largest group
    largest_group = groups[0]
    for i in range(1, len(groups)):
        if len(largest_group) < len(groups[i]):
            largest_group = groups[i]

    if debug_info is not None:
        debug_info.largest_group = largest_group

    # Remove the octagon if it was detected
    max_area = 0
    largest = 0
    for ind, i in enumerate(largest_group):
        area = cv2.contourArea(filtered_contours[i])
        if areas[i] > max_area:
            max_area = area
            largest = ind

    if len(filtered_contours[largest_group[largest]]) == 8:
        del largest_group[largest]

    return filtered_contours, largest_group


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def circlify(contour):
    moments = cv2.moments(contour)
    center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
    distance = np.sqrt(np.sum((contour[:, 0, :] - center) ** 2, axis=1))
    mean, stddev = cv2.meanStdDev(distance)

    close_enough = abs(distance - mean[0, 0]) < stddev[0, 0]
    filtered_contour = contour[close_enough, :, :]
    # print("Full %d -> Filtered %d" % (len(contour), len(filtered_contour)))
    moments = cv2.moments(filtered_contour)
    center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
    distance = np.sqrt(np.sum((contour[:, 0, :] - center) ** 2, axis=1))
    mean, stddev = cv2.meanStdDev(distance[close_enough])

    close_enough = abs(distance - mean[0, 0]) < 3 * stddev[0, 0]
    for i in range(len(close_enough)):
        if close_enough[i]:
            continue
        contour[i, 0, :] = center + (contour[i, 0, :] - center) * mean / distance[i]

    moments = cv2.moments(contour)
    center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
    distance = np.sqrt(np.sum((filtered_contour[:, 0, :] - center) ** 2, axis=1))
    mean = np.mean(distance)

    return center, mean


def find_outer_circle(image, debug_info=None):
    # Downsample for better performance
    downsampled_image = cv2.pyrDown(image)
    edges = cdm(downsampled_image, auto_threshold_multiplier=0.75)
    _, contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if debug_info is not None:
        debug_info.image = downsampled_image
        debug_info.edge_image = edges
        debug_info.contours = contours

    filtered_contours, largest_group = __find_circles(contours, 1)

    if filtered_contours is None or len(largest_group) < 3:
        return None, None

    largest = None
    largest_area = 0
    for ind in largest_group:
        area = cv2.contourArea(filtered_contours[ind])
        if area > largest_area:
            largest_area = area
            largest = filtered_contours[ind]
    rect = cv2.boundingRect(largest)
    margin = 10
    sub_image = image[rect[1]-margin:rect[1]+rect[3]+margin, rect[0]-margin:rect[0]+rect[2]+margin, :]

    edges = cdm(sub_image, auto_threshold_multiplier=1.0)
    _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if debug_info is not None:
        debug_info.sub_image = sub_image
        debug_info.sub_edge_image = edges
        debug_info.sub_contours = contours
    largest_area = 0
    largest = None
    for i in range(len(contours)):
        approx = cv2.approxPolyDP(contours[i], 0.005 * cv2.arcLength(contours[i], True), True)
        if len(approx) == 8:
            continue
        area = cv2.contourArea(contours[i])
        if area > largest_area:
            largest_area = area
            largest = contours[i]
    #center, radius = cv2.minEnclosingCircle(largest)
    center, radius = circlify(largest)
    center = (center[0] + rect[0] - margin, center[1] + rect[1] - margin)

    return  center, radius
