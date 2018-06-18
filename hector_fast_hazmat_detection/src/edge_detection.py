import cv2
import numpy as np


def edgedetect (channel):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)

    sobel[sobel > 255] = 255  # Some values seem to go above 255. However RGB channels has to be within 0-255
    return sobel


def cdm(image, image_mem, threshold=None):
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
        std = np.var(max_response[max_response > 0])
        threshold = mean # + 0.05 * std
    _, max_response = cv2.threshold(max_response, threshold, 255, cv2.CV_8U)
    return max_response


def color_edges(image):
    im = image.astype(np.float) / 255

    y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    x_filter = y_filter.T

    rx = cv2.filter2D(image[:, :, 0], cv2.CV_32F, x_filter)
    gx = cv2.filter2D(image[:, :, 1], cv2.CV_32F, x_filter)
    bx = cv2.filter2D(image[:, :, 2], cv2.CV_32F, x_filter)

    ry = cv2.filter2D(image[:, :, 0], cv2.CV_32F, y_filter)
    gy = cv2.filter2D(image[:, :, 1], cv2.CV_32F, y_filter)
    by = cv2.filter2D(image[:, :, 2], cv2.CV_32F, y_filter)

    Jx = rx ** 2 + gx ** 2 + bx ** 2
    Jy = ry ** 2 + gy ** 2 + by ** 2
    Jxy = rx * ry + gx * gy + bx * by

    # compute first (greatest) eigenvalue of 2x2 matrix J'*J.
    # note that the abs() is only needed because some values may be slightly
    # negative due to round-off error.
    D = np.sqrt(np.abs(Jx ** 2 - 2 * Jx * Jy + Jy ** 2 + 4 * Jxy ** 2))
    e1 = (Jx + Jy + D) / 2
    # the 2nd eigenvalue would be:  e2 = (Jx + Jy - D) / 2

    edge_magnitude = np.sqrt(e1)

    # compute edge orientation (from eigenvector tangent)
    edge_orientation = np.arctan2(-Jxy, e1 - Jy)
    edge_orientation = edge_orientation * 180 / np.pi
    return (edge_magnitude * 255 / np.max(edge_magnitude)).astype(np.uint8), edge_orientation


def is_smaller(img, x1, y1, x2, y2):
    if x2 < 0 or x2 >= img.shape[0]:
        return False
    if y2 < 0 or y2 >= img.shape[1]:
        return False
    return img[x1, y1] < img[x2, y2]


def non_max_suppression(edges, orientation):
    out = np.empty_like(edges)
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if orientation[i, j] > 62.5 or orientation[i, j] < -62.5:
                zero = is_smaller(edges, i, j, i, j - 1) or is_smaller(edges, i, j, i, j + 1)
            elif orientation[i, j] > 22.5:
                zero = is_smaller(edges, i, j, i + 1, j + 1) or is_smaller(edges, i, j, i - 1, j - 1)
            elif orientation[i, j] > -22.5:
                zero = is_smaller(edges, i, j, i + 1, j) or is_smaller(edges, i, j, i - 1, j)
            else:
                zero = is_smaller(edges, i, j, i - 1, j + 1) or is_smaller(edges, i, j, i + 1, j - 1)
            out[i, j] = 0 if zero else edges[i, j]
    return out


def calculate_threshold(edges):
    oldmean = 0
    mean = np.mean(edges)
    while np.abs(mean - oldmean) > 0.1:
        oldmean = mean
        high = np.mean(edges[edges >= mean])
        low = np.mean(edges[edges < mean])
        mean = (high + low) / 2
    return mean


def calculate_thresholds(edges):
    oldupper = 0
    oldlower = 0
    upper = 160
    lower = 80
    while np.abs(upper - oldupper) > 0.1 or np.abs(lower - oldlower) > 0.1:
        oldupper = upper
        oldlower = lower
        high_mean = np.mean(edges[edges > upper])
        middle_mean = np.mean(edges[np.logical_and(lower < edges, edges <= upper)])
        low_mean = np.mean(edges[edges <= lower])

        upper = (high_mean + middle_mean) / 2
        lower = (middle_mean + low_mean) / 2
    return upper, lower


# def flowThreshold(edges, out, lower, row, col):


def threshold(edges, upper, lower):
    highs = []
    indicator = np.zeros_like(edges, dtype=np.uint8)
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i, j] >= upper:
                highs.append((i, j))
                indicator[i, j] = 255
    while len(highs) > 0:
        pt = highs[-1]
        del highs[-1]
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                row = pt[0] + i
                col = pt[1] + j
                if row == -1 or row == edges.shape[0] or col == -1 or col == edges.shape[1]:
                    continue
                if indicator[row, col] != 0:
                    continue
                if edges[row, col] < lower:
                    edges[row, col] = 0
                    continue
                indicator[row, col] = 255
                highs.append((row, col))
    #edges[np.logical_and(indicator == 0, edges < upper)] = 0
    return indicator


def extract_foreground_mask(image):
    gpu = isinstance(image, cv2.UMat)
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    image_lab_mem = image_lab.get() if gpu else image_lab
    mean = np.mean(image_lab_mem.reshape((image_lab_mem.shape[0] * image_lab_mem.shape[1], 3)), axis=0)
    diff = image_lab_mem - mean
    diff = np.sum(diff[:, :, 1:] ** 2, axis=2)
    diff = diff * 255 / np.max(diff)
    diff = np.round(diff)
    diff[diff < np.mean(diff)] = 0
    diff[diff > 0] = 255
    diff = diff.astype(np.uint8)
    if gpu:
        diff = cv2.UMat(diff)

    diff = cv2.dilate(cv2.erode(diff, None), None, iterations=2)
    return diff


def lab_edge_detection(image):
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    image_lab = image_lab.get() if isinstance(image_lab, cv2.UMat) else image_lab

    n = 65
    big_filter = np.ones((n, n))
    big_filter = -big_filter
    big_filter[n // 2, n // 2] = n * n - 1

    # Remove L channel
    response = cv2.filter2D(image_lab[:, :, 1:], cv2.CV_32F, big_filter)
    binary = np.sum(response**2, axis=2)
    binary = binary * 255 / np.max(binary)
    binary = binary.astype(np.uint8)
    mean = cv2.mean(binary)[0]
    _, binary = cv2.threshold(binary, mean, 255, cv2.THRESH_BINARY)
    binary = cv2.dilate(cv2.erode(binary, None), None)
    edges = cv2.Canny(binary, 20, 1)
    return edges
