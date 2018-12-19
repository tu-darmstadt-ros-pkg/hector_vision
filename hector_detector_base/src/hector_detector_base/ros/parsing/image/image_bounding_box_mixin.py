import cv2
import numpy as np
from . import ImageMixin


class ImageBoundingBoxMixin(ImageMixin):
    def __init__(self, *args, **kwargs):
        super(ImageBoundingBoxMixin, self).__init__(*args, **kwargs)
        if 'bounding_box_color' in kwargs:
            self.bounding_box_color = kwargs['bounding_box_color']
        elif 'color' in kwargs:
            self.bounding_box_color = kwargs['color']
        else:
            self.bounding_box_color = (255, 0, 0)
        if 'bounding_box_thickness' in kwargs:
            self.bounding_box_thickness = kwargs['bounding_box_thickness']
        elif 'thickness' in kwargs:
            self.bounding_box_thickness = kwargs['thickness']
        else:
            self.bounding_box_thickness = 2
        if 'bounding_box_contour_fallback' in kwargs:
            self.bounding_box_contour_fallback = kwargs['bounding_box_contour_fallback']
        else:
            self.bounding_box_contour_fallback = True

    def fill_image(self, msg, image, detection):
        super(ImageBoundingBoxMixin, self).fill_image(msg, image, detection)
        if detection.bounding_rect is not None:
            x, y, w, h = detection.bounding_rect
        elif self.bounding_box_contour_fallback and detection.contours is not None:
            x, y, w, h = cv2.boundingRect(np.concatenate(detection.contours))
        else:
            return
        x += detection.x_offset
        y += detection.y_offset
        cv2.rectangle(image, (x, y), (x+w, y+h), self.bounding_box_color, self.bounding_box_thickness)
