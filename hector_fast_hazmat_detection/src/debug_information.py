import numpy as np


class DebugInformation:
    approx_contours = []  # type: List[np.ndarray]
    best_matches = []  # type: List[(np.ndarray, np.ndarray, np.ndarray, float)]
    contours = []  # type: List[np.ndarray]
    detections = []  # type: List[(np.ndarray, np.ndarray, np.ndarray, float)]
    dilated_image = None  # type: np.ndarray
    edge_image = None  # type: np.ndarray
    filtered_contours = []  # type: List[np.ndarray]
    input_image = None  # type: np.ndarray
    matches = []
    regions_of_interest = None  # type: List[tuple]
    rejections = []
    sub_images = []  # type: List[(np.ndarray, np.ndarray, np.ndarray)]
    signs = []  # type: List

    def __init__(self):
        pass
