from ...mixin_base import MixinBase


class DetectionMessageMixin(MixinBase):
    def __init__(self, *args, **kwargs):
        super(DetectionMessageMixin, self).__init__(*args, **kwargs)

    def fill_detection_message(self, msg, detection_message, detection):
        """
        Fills the given detection message with values
        :param msg: The input message received by the detector
        :param detection_message: The detection message whose properties are set
        :type detection_message: hector_detection_msgs.msg.Detection
        :param detection: A detection by the detector
        :type detection: hector_detector_base.ImageDetection
        """
        pass
