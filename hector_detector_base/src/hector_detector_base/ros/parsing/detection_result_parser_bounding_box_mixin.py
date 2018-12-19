from geometry_msgs.msg import Point32, Polygon
from .detection_message_mixin import DetectionMessageMixin


def make_point_message(x, y):
    pt = Point32()
    pt.x = x
    pt.y = y
    return pt


class DetectionResultParserBoundingBoxMixin(DetectionMessageMixin):
    def __init__(self, *args, **kwargs):
        super(DetectionResultParserBoundingBoxMixin, self).__init__(*args, **kwargs)

    def fill_detection_message(self, msg, detection_message, detection):
        super(DetectionResultParserBoundingBoxMixin, self).fill_detection_message(msg, detection_message, detection)
        if detection.bounding_rect is None:
            return
        detection_message.x_offset = detection.x_offset
        detection_message.y_offset = detection.y_offset
        msg_contour = Polygon()
        msg_contour.points.append(make_point_message(detection.bounding_rect[0], detection.bounding_rect[1]))
        msg_contour.points.append(make_point_message(detection.bounding_rect[0] + detection.bounding_rect[2],
                                                     detection.bounding_rect[1]))
        msg_contour.points.append(make_point_message(detection.bounding_rect[0] + detection.bounding_rect[2],
                                                     detection.bounding_rect[1] + detection.bounding_rect[3]))

        msg_contour.points.append(make_point_message(detection.bounding_rect[0],
                                                     detection.bounding_rect[1] + detection.bounding_rect[3]))
        detection_message.contours.append(msg_contour)
