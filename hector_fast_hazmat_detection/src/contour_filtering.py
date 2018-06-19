import cv2
import numpy as np
from distance_measures import euclidean_distance


def approximate_contour(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.08 * peri, True)
    return approx, peri


def is_square(contour):
    if len(contour) != 4:
        return False

    ad = euclidean_distance(contour[-1, 0, :], contour[0, 0, :])
    ab = euclidean_distance(contour[0, 0, :], contour[1, 0, :])
    if np.abs(ad - ab) / np.maximum(ad, ab) > 0.2:
        return False
    bc = euclidean_distance(contour[1, 0, :], contour[2, 0, :])
    if np.abs(ab - bc) / np.maximum(ab, bc) > 0.2:
        return False
    cd = euclidean_distance(contour[2, 0, :], contour[3, 0, :])
    return np.abs(bc - cd) / np.maximum(bc, cd) <= 0.2 and np.abs(cd - ad) / np.maximum(cd, ad) <= 0.2


def is_rectangle(contour):
    if len(contour) != 4:
        return False

    ad = euclidean_distance(contour[-1, 0, :], contour[0, 0, :])
    bc = euclidean_distance(contour[1, 0, :], contour[2, 0, :])
    if np.abs(ad - bc) / np.maximum(ad, bc) > 0.25:
        return False
    ab = euclidean_distance(contour[0, 0, :], contour[1, 0, :])
    cd = euclidean_distance(contour[2, 0, :], contour[3, 0, :])
    return np.abs(ab - cd) / np.maximum(ad, cd) <= 0.25


def filter_large_contours(contours):
    filtered_contours = []
    filtered_areas = []
    for c in contours:
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        if area < 1000 or area > 300 ** 2:
            continue
        filtered_contours.append(c)
        filtered_areas.append(area)
    return filtered_contours, filtered_areas


def filter_contours(contours):
    filtered_contours = []
    for c in contours:
        approx, peri = approximate_contour(c)
        # if not is_square(approx):
        #if len(approx) < 4:
        #    continue
        area = cv2.contourArea(approx)
        if ((len(approx) == 3 and 600 < area and 0.2 < area / (peri / 4)**2) or
            (900 < area and 0.4 < area / (peri / 4)**2)):
            filtered_contours.append(approx)
    return filtered_contours
