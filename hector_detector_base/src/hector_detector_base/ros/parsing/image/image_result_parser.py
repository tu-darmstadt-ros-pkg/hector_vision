from cv_bridge import CvBridge
from .... import ResultParser
from ....image_helper import image_from_sensor_message
from .image_mixin import ImageMixin


class ImageResultParser(ResultParser, ImageMixin):
    def __init__(self, *args, **kwargs):
        super(ImageResultParser, self).__init__(*args, **kwargs)
        self.cv_bridge = CvBridge()

        self.encoding = kwargs['encoding'] if 'encoding' in kwargs else None
        self.no_alpha = kwargs['no_alpha'] if 'no_alpha' in kwargs else False

    def parse(self, msg, detection_result):
        image = image_from_sensor_message(self.cv_bridge, msg, self.encoding, self.no_alpha)
        for detection in detection_result.detections:
            self.fill_image(msg, image, detection)
        return self.cv_bridge.cv2_to_imgmsg(image, self.encoding)
