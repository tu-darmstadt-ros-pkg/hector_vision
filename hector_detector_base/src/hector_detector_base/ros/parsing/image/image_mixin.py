import numpy as np
from ....mixin_base import MixinBase
from ....image_detection import ImageDetection


class ImageMixin(MixinBase):
    def __init__(self, *args, **kwargs):
        super(ImageMixin, self).__init__(*args, **kwargs)

    def fill_image(self, msg, image, detection):
        """
        Fills the image with data from the detection.
        :param msg: The input message of the detector
        :param image: The image that is filled
        :type image: np.ndarray
        :param detection: The detection made by the detector
        :type detection: ImageDetection
        """
        pass
