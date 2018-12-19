

def image_from_sensor_message(cv_bridge, msg, encoding=None, no_alpha=False):
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
    if encoding is None:
        if msg.encoding == "bgr8" or msg.encoding == "bgr16":
            encoding = "bgr8"
        elif msg.encoding == "rgb8" or msg.encoding == "rgb16":
            encoding = "rgb8"
        elif msg.encoding == "mono8":
            encoding = "mono8"
        elif msg.encoding == "mono16":
            encoding = "mono16"
        elif msg.encoding == "bgra8" or msg.encoding == "bgra16":
            encoding = "bgra8" if not no_alpha else "bgr8"
        elif msg.encoding == "rgba8" or msg.encoding == "rgba16":
            encoding = "rgba8" if not no_alpha else "rgb8"
        else:
            raise NotImplementedError("Unknown format '{}'. Pls fix.".format(msg.encoding))
    return cv_bridge.imgmsg_to_cv2(msg, encoding)