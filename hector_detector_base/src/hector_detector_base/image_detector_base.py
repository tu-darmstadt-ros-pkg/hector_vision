from numpy import ndarray
from cv_bridge import CvBridge
import sensor_msgs.msg
from typing import Optional
from .cancellation_token import CancellationToken
from .image_detection_result import ImageDetectionResult
from .image_helper import image_from_sensor_message


class ImageDetectorConfiguration(object):
    def __init__(self, cache_enabled=True, encoding=None):
        """
        :param cache_enabled: Whether or not the last result should be cached.
         (Cached results can be retrieved using has_result and get_result)
        :param encoding: Can be used to force conversion to a specific encoding. Set to none to determine from sensor
         msg
        """
        self.cache_enabled = cache_enabled
        self.encoding = None


class ImageDetectorBase(object):
    def __init__(self, detector_configuration=None):
        """
        :param detector_configuration: The configuration for the detector
        :type detector_configuration: ImageDetectorConfiguration
        """
        self.configuration = detector_configuration or ImageDetectorConfiguration()
        self._last_result = None
        self.cv_bridge = CvBridge()

    def has_result(self):
        return self._last_result is not None

    def get_result(self):
        return self._last_result

    def get_image_from_sensor_message(self, msg, encoding=None, no_alpha=False):
        """
        Converts a sensor_msgs.msg.Image to a cv image (numpy.ndarray).
        Optionally, the encoding of the returned image can be forced by setting encoding to one of the following str
        values:
         - rgb8
         - rgba8
         - bgr8
         - bgra8
         - mono8
         - mono16
         If the encoding is None, the encoding is determined from the image message.
         The parameter no_alpha only applies if the encoding is automatically determined and will convert input images
         using rgba (or bgra) to rgb (bgr) instead of rgba (bgra).
        :param msg: The input sensor message.
        :type msg: sensor_msgs.msg.Image
        :param encoding: (Optional) Forces encoding in a specific target format.
        :type encoding: Optional[str]
        :param no_alpha: (Optional) When automatically determining the target encoding, will disregard whether the input
         had an alpha channel or not.
        :type no_alpha: bool
        :return: The converted image
        :rtype: ndarray
        """
        return image_from_sensor_message(self.cv_bridge, msg, encoding, no_alpha)

    def internal_detect(self, image, cancellation_token=None):
        """
        Internal detection method that is implemented in subclasses.
        :param image: The input image
        :type image: sensor_msgs.msg.Image
        :param cancellation_token: (Optional) A token that is used to cancel the detection.
        :type cancellation_token: Optional[CancellationToken]
        :return: The result of the detection
        :rtype: ImageDetectionResult
        """
        raise NotImplementedError("Has to be implemented in subclass!")

    def detect(self, image, cancellation_token=None):
        """
        Runs a detection on a given image

        :param image: The input image for the detector
        :type image: sensor_msgs.msg.Image
        :param cancellation_token: (Optional) A token that can be used to cancel the detection.
         May not be supported by the detector.
        :type cancellation_token: Optional[CancellationToken]
        :rtype: ImageDetectionResult
        :return: What the detector detected in the given image.
        """
        result = self.internal_detect(image, cancellation_token)
        if self.configuration.cache_enabled:  # TODO: How to handle a cancelled result?
            self._last_result = result
        return result

