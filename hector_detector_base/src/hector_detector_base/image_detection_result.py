import numpy as np
from typing import List
from .image_detection import ImageDetection


class ImageDetectionResult(object):
    """
    A detection result for an image including all separate detections in that image.

    Attributes:
        detections      A list of detections in the given image
        debug_images    If debug mode is enabled in the configuration, this field can be used to store debug images
                         relevant to all detections
        cancelled       Indicating whether the detector was cancelled before it finished
    """
    def __init__(self):
        self.detections = []  # type: List[ImageDetection]
        self.debug_images = []  # type: List[np.ndarray]
        self.cancelled = False
