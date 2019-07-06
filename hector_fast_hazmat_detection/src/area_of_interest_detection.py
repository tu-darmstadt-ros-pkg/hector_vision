import cv2
import numpy as np
from debug_information import DebugInformation
import hector_vision
from distance_measures import *
from contour_filtering import *
from edge_detection import *


def is_square(contour):
    if len(contour) != 4:
        return False
    shortest_side = squared_distance(contour[-1, 0], contour[0, 0])
    longest_side = shortest_side
    for i in range(3):
        length = squared_distance(contour[i, 0], contour[i + 1, 0])
        if length < shortest_side:
            shortest_side = length
        elif length > longest_side:
            longest_side = length
    print((longest_side - shortest_side) / float(longest_side))
    return (longest_side - shortest_side) / float(longest_side) > 0.8


def detect_areas_of_interest(image, downsample_passes=1, debug_info=None):
    image_mem = image.get() if isinstance(image, cv2.UMat) else image
    width = image_mem.shape[1]
    height = image_mem.shape[0]
    original_width = width
    original_height = height
    image = cv2.resize(image, None, None, 0.5**downsample_passes, 0.5**downsample_passes, cv2.INTER_AREA)
    width = width // 2**downsample_passes
    height = height // 2**downsample_passes

    img_edges, orientation = hector_vision.color_edges(image)
    upper, lower = hector_vision.calculate_thresholds(img_edges)
    #mask = hector_vision.threshold(img_edges, upper, lower)
    #img_edges[mask == 0] = lower
    #upper, lower = hector_vision.calculate_thresholds(img_edges)
    img_edges = hector_vision.threshold(img_edges, upper, lower)

    img_edges = cv2.dilate(img_edges, None)
    img_edges = cv2.erode(img_edges, None)
    img_edges = cv2.dilate(img_edges, None)

    _, contours, _ = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        connected_to_wall = False
        for pt in c[:, 0, :]:
            if pt[1] < 4 or pt[1] + 4 >= img_edges.shape[0]:
                connected_to_wall = True
                break
        if not connected_to_wall:
            continue
        cv2.fillConvexPoly(img_edges, c, 0)

    _, contours, _ = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        c *= 2**downsample_passes
    if debug_info is not None:
        debug_info.edge_image = img_edges.get() if isinstance(img_edges, cv2.UMat) else img_edges
        debug_info.contours = contours
    contours, areas = filter_large_contours(contours)
    filtered_contours = filter_contours(contours)

    regions_of_interest = []
    margin = 16
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        right = x + w + margin
        bottom = y + h + margin
        x = x - margin if x >= margin else 0
        y = y - margin if y >= margin else 0
        w = right - x if right <= original_width else original_width - x
        h = bottom - y if bottom <= original_height else original_height - y

        for i in range(len(regions_of_interest) - 1, -1, -1):
            roi = regions_of_interest[i]
            if roi[0] + roi[2] > x and x + w > roi[0] and roi[1] + roi[3] > y and y + h > roi[1]:
                x2, y2, w2, h2 = roi
                del regions_of_interest[i]
                right = np.maximum(x+w, x2+w2)
                bottom = np.maximum(y+h, y2+h2)
                x = np.minimum(x, x2)
                y = np.minimum(y, y2)
                w = right - x
                h = bottom - y
        regions_of_interest.append((x, y, w, h))
    for i in range(len(regions_of_interest)-1, -1, -1):
        roi = regions_of_interest[i]
        if roi[0] < 5 or roi[1] < 5:
            del regions_of_interest[i]
        elif roi[0] + roi[2] + 5 > original_width or roi[1] + roi[3] + 5 > original_height:
            del regions_of_interest[i]
    if debug_info is not None:
        debug_info.filtered_contours = filtered_contours
        debug_info.regions_of_interest = regions_of_interest

    return filtered_contours, regions_of_interest
