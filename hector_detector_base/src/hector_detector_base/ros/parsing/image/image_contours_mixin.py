import cv2
import numpy as np
from . import ImageMixin


class ImageContoursMixin(ImageMixin):
    def __init__(self, *args, **kwargs):
        super(ImageContoursMixin, self).__init__(*args, **kwargs)

        if 'contour_color' in kwargs:
            self.contour_color = kwargs['contour_color']
        elif 'color' in kwargs:
            self.contour_color = kwargs['color']
        else:
            self.contour_color = (255, 0, 0)
        if 'contour_thickness' in kwargs:
            self.contour_thickness = kwargs['contour_thickness']
        elif 'thickness' in kwargs:
            self.contour_thickness = kwargs['thickness']
        else:
            self.contour_thickness = None

    def fill_image(self, msg, image, detection):
        super(ImageContoursMixin, self).fill_image(msg, image, detection)
        contours = [c + np.array([detection.x_offset, detection.y_offset]) for c in detection.contours]
        cv2.drawContours(image, contours, -1, self.contour_color, self.contour_thickness)
