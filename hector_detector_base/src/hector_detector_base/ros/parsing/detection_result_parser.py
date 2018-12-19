import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point32, Polygon
from hector_detection_msgs.msg import Detection, DetectionResult
from ... import ResultParser, ImageDetectionResult
from .detection_message_mixin import DetectionMessageMixin


def make_point_message(x, y):
    pt = Point32()
    pt.x = x
    pt.y = y
    return pt


class DetectionResultParser(ResultParser, DetectionMessageMixin):
    def __init__(self, *args, **kwargs):
        super(DetectionResultParser, self).__init__(*args, **kwargs)

    def fill_detection_message(self, msg, detection_message, detection):
        detection_message.object_class = detection.object_class
        detection_message.name = detection.name
        detection_message.value = detection.value
        detection_message.data = detection.data
        detection_message.support = detection.support

    def parse(self, msg, detection_result):
        """
        :param msg: The input message
        :param detection_result: The result of the detector
        :type detection_result: ImageDetectionResult
        :returns: A hector_detection_msgs/DetectionResult message with all fields available are set
        :rtype. DetectionResult
        """
        msg_result = DetectionResult()
        for detection in detection_result.detections:
            msg_detection = Detection()
            self.fill_detection_message(msg, msg_detection, detection)
            msg_result.detections.append(msg_detection)
        return msg_result
