import numpy as np


def manhattan_distance(a, b):
    return np.sum(np.abs(a-b))


def squared_distance(a, b):
    return np.sum((a - b)**2)


def euclidean_distance(a, b):
    return np.sqrt(squared_distance(a, b))


def get_distance_of_contours(a, b):
    smallest_distance = 10000000
    for pt1 in a:
        for pt2 in b:
            dist = manhattan_distance(pt1[0], pt2[0])
            if dist < smallest_distance:
                smallest_distance = dist
    return smallest_distance
