import numpy as np
from typing import List


class ImageDetection(object):
    """
    A detection in an image.

    Attributes:
        x_offset        The x offset (horizontal axis) of the sub image, contour and bounding rect in the original image
        y_offset        The y offset (vertical axis) of the sub image, contour and bounding rect in the original image
        contours        The contour(s) of the detected object(s) in the sub image
        bounding_rect   The bounding rect of the detected object in the sub image. (x, y, w, h)
        object_class    The detected object's class, e.g., 'hazmat_sign' if a hazmat sign was detected
        name            The detected object's name, e.g., 'spontaneously_combustible' to stay within the hazmat domain
        data            Associated string data of any kind, e.g., for a qr code detection this field may contain the value of the qr code
        value           Associated float data, e.g., a manometer detection may use this field to store the detected value
        image           The sub image where the detection took place
        support         TODO
        debug_images    If debug mode is enabled in the configuration, this field can be used to store debug images related to this specific detection

    """
    def __init__(self):
        self.x_offset = 0  # type: int
        self.y_offset = 0  # type: int
        self.contours = None  # type: List[np.ndarray]
        self.bounding_rect = None  # type: (int, int, int, int)
        self.object_class = ''  # type: str
        self.name = ''  # type: str
        self.data = ''  # type: str
        self.value = np.nan  # type: float
        self.image = None  # type: np.ndarray
        self.support = np.nan  # type: float
        self.debug_images = []  # type: List[np.ndarray]
