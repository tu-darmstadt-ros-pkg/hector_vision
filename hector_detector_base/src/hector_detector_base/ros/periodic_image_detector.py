import rospy
import sensor_msgs.msg
from time import time
from typing import List

from .. import ImageDetectorBase
from .result_publisher import ResultPublisher


class PeriodicImageDetector(object):

    def __init__(self, detector, input_topic="~/image", rate=1):
        """
        :param detector: The detector that is run periodically.
        :type detector: ImageDetectorBase
        :param input_topic: The topic on which the input image is published.
        :type input_topic: str
        :param rate: Hz rate how often the detector runs. Pass -1 to run whenever a new image is received.
        :type rate: float
        """
        self.detector = detector
        self.input_topic = input_topic
        self.rate = rate
        self.publishers = []  # type: List[ResultPublisher]
        self.image_subscriber = rospy.Subscriber(input_topic, sensor_msgs.msg.Image, self.image_callback)

        self.last_msg = None
        self.last_execution = 0

    def add_result_publisher(self, publisher):
        """
        Adds a result publisher to the end of the list.
        Result publishers are published in order.

        :param publisher: The publisher that is appended.
        :type publisher: ResultPublisher
        :return: None
        """
        self.publishers.append(publisher)

    def remove_result_publisher(self, publisher):
        """
        Removes a result publisher from the list of result publishers.

        :param publisher: The publisher that is to be removed.
        :type publisher: ResultPublisher
        :return: None
        """
        self.publishers.remove(publisher)

    def image_callback(self, msg):  # type: (sensor_msgs.msg.Image) -> None
        self.last_msg = msg
        if self.rate == -1 and len(self.publishers) > 0:
            result = self.detector.detect(msg)
            self._publish_result(msg, result)
            self.last_msg = None

    def run(self):
        """
        Will periodically run the detector with the specified rate until application is shutdown.
        """
        run_detections = self.rate > 0
        rate = rospy.Rate(self.rate) if run_detections else rospy.Rate(1)
        while not rospy.is_shutdown():
            if run_detections:
                self.run_once(True)
            try:
                rate.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                pass

    def run_once(self, force_detection=False):
        """
        Runs the detector once if the time specified by the rate has passed.
        The time can be ignored using force_detection, however, the detector will only run if a new image was received.

        :param force_detection: If True, ignores how much time has passed and run the detector. Default: False
        """
        if self.last_msg is None or len(self.publishers) == 0:
            return
        execution_time = time()
        if not force_detection and (execution_time - self.last_execution) < 1 / self.rate:
            return
        self.last_execution = execution_time
        image = self.last_msg
        result = self.detector.detect(image)
        self.last_msg = None
        self._publish_result(self.last_msg, result)

    def _publish_result(self, msg, result):
        for pub in self.publishers:
            pub.publish(msg, result)
