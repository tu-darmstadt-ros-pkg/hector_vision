from geometry_msgs.msg import Point32, Polygon
from .detection_message_mixin import DetectionMessageMixin


def make_point_message(x, y):
    pt = Point32()
    pt.x = x
    pt.y = y
    return pt


class DetectionResultParserContoursMixin(DetectionMessageMixin):
    def __init__(self, *args, **kwargs):
        super(DetectionResultParserContoursMixin, self).__init__(*args, **kwargs)

    def fill_detection_message(self, msg, detection_message, detection):
        super(DetectionResultParserContoursMixin, self).fill_detection_message(msg, detection_message, detection)
        if detection.contours is None:
            return
        detection_message.x_offset = detection.x_offset
        detection_message.y_offset = detection.y_offset
        for contour in detection.contours:
            msg_contour = Polygon()
            for pt in contour:
                msg_contour.points.append(make_point_message(pt[0][0], pt[0][1]))
            detection_message.contours.append(msg_contour)
