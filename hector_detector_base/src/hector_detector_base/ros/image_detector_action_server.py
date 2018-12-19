from actionlib import SimpleActionServer
from geometry_msgs.msg import Point32, Polygon
from .. import CancellationToken, ImageDetectorBase, ResultParser
import hector_detection_msgs.msg
from parsing.detection_result_parser import DetectionResultParser


class ImageDetectorActionServer(object):
    def __init__(self, topic, detector, detection_result_parser=DetectionResultParser()):
        """
        Creates an action server that can be reached on the given topic and runs the given detector.
        Has to be started using the start() method.

        :param topic: The topic this detection action server is running on
        :type topic: str
        :param detector: The detector that runs when the action is executed
        :type detector: ImageDetectorBase
        :param detection_result_parser: A parser that gets the detection result and returns a
         hector_detection_msgs.msg.DetectionResult message
        :type detection_result_parser: ResultParser
        """
        self.detector = detector
        self.action_server = SimpleActionServer(topic, hector_detection_msgs.msg.DetectObjectAction,
                                                execute_cb=self.execute, auto_start=False)
        self.action_server.register_preempt_callback(self.preempt)
        self.detection_result_parser = detection_result_parser

        self.goal = None
        self.cancellation_token = None  # type: CancellationToken

    def start(self):
        self.action_server.start()

    def preempt(self):
        if self.cancellation_token is not None:
            self.cancellation_token.cancel()

    def on_preempt(self, goal, result):
        self.action_server.set_preempted()

    def on_succeeded(self, goal, result):
        msg_result = hector_detection_msgs.msg.DetectObjectResult()
        msg_result.success = True
        msg_result.detection_result = self.detection_result_parser.parse(goal.image, result)
        self.action_server.set_succeeded(result)

    def execute(self, goal):  # type: (hector_detection_msgs.msg.DetectObjectGoal) -> None
        self.cancellation_token = CancellationToken()
        try:
            result = self.detector.detect(goal.image, self.cancellation_token)
        except:
            print("Exception led to action server dieing.")
            self.action_server.set_aborted(text="Exception occurred!")
            raise
        if self.cancellation_token.cancellation_requested:
            self.on_preempt(goal, result)
        else:
            self.on_succeeded(goal, result)
