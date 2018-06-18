import numpy as np


class DebugInformation:
    best_matches = None  # type: List[(np.ndarray, np.ndarray, np.ndarray, float)]
    detections = None  # type: List[(np.ndarray, np.ndarray, np.ndarray, float)]
    sub_images = None  # type: List[(np.ndarray, np.ndarray, np.ndarray)]
    filtered_contours = None  # type: List[np.ndarray]
    regions_of_interest = None # type: List[tuple]
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