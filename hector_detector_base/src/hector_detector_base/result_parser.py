from .mixin_base import MixinBase


class ResultParser(MixinBase):
    def __init__(self, *args, **kwargs):
        super(ResultParser, self).__init__(*args, **kwargs)

    def parse(self, msg, detection_result):
        """
        :param msg: The input message
        :param detection_result: The result of the detector
        :type detection_result: ImageDetectionResult
        :returns: A different interpretation of the result, e.g. a message with the detections that can be published.
        """
        raise NotImplementedError("Has to be implemented in subclass!")
