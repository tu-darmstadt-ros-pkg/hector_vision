from cv_bridge import CvBridge
from .image_mixin import ImageMixin
from ..detection_message_mixin import DetectionMessageMixin


class DetectionResultParserImageMixin(DetectionMessageMixin, ImageMixin):
    def __init__(self, *args, **kwargs):
        super(DetectionResultParserImageMixin, self).__init__(*args, **kwargs)
        self.cv_bridge = CvBridge()

    def fill_detection_message(self, msg, detection_message, detection):
        super(DetectionResultParserImageMixin, self).fill_detection_message(msg, detection_message, detection)
        if detection.image is None:
            return
        image = detection.image.copy()
        # For the drawing set the offsets to zero because they are used by the mixins
        x_offset = detection.x_offset
        y_offset = detection.y_offset
        detection.x_offset = 0
        detection.y_offset = 0
        self.fill_image(msg, image, detection)
        detection.x_offset = x_offset
        detection.y_offset = y_offset
        detection_message.sub_image = self.cv_bridge.cv2_to_imgmsg(image)
