from rospy import Publisher
from .. import ResultParser, ImageDetectionResult


class ResultPublisher(object):
    def __init__(self, topic, msg_type, parser):
        """
        :param topic: The topic the publisher published to.
        :type topic: str
        :param msg_type: The message class for serialization
        :param parser: A parser that converts the detection result to the topic message type.
        :type parser: ResultParser
        """
        self.topic = topic
        self.parser = parser
        self.perception_pub = Publisher(topic, msg_type, queue_size=10)

    def publish(self, msg, detection_result):
        """
        Publishes the result of a detection

        :param msg: The input message
        :param detection_result: The detection result.
        :type detection_result: ImageDetectionResult
        :return: None
        """
        self.perception_pub.publish(self.parser.parse(msg, detection_result))
