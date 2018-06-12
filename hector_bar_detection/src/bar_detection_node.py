#!/usr/bin/env python
from __future__ import print_function, division
import rospy
import cv_bridge

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from hector_perception_msgs.msg import LocalizeBarsAction
from hector_perception_msgs.msg import LocalizeBarsResult

from enum import Enum

import sensor_msgs.msg
import geometry_msgs.msg
import hector_perception_msgs.msg
import hector_perception_msgs.srv
import hector_nav_msgs.srv
import bar_detection
import actionlib


class BarDetectionErrorType(Enum):
    NoError = 0
    NoImageError = 1
    ProjectPixelTo3DRayError = 2
    GetDistanceToObstacleError = 3
    NoIntersectionPointError = 4


class BarDetectionError(Exception):
    def __init__(self, message, error):
        super(BarDetectionError, self).__init__(message)
        if not isinstance(error, BarDetectionErrorType):
            raise TypeError("Error must be set to a BarDetectionErrorType.")
        self.error = error


class BarDetectionNode:
    def __init__(self):
        self.last_front_image = None
        self.last_back_image = None
        self.bridge = cv_bridge.CvBridge()
        self.detector = bar_detection.BarDetection()
        self.detection_image_pub = rospy.Publisher("~detection_image", sensor_msgs.msg.Image, queue_size=10, latch=True)
        self.perception_pub = rospy.Publisher("image_percept", hector_perception_msgs.msg.PerceptionDataArray,
                                              queue_size=10)
        self.debug = rospy.get_param("~debug", False)
        self.front_image_sub = rospy.Subscriber("~front_image", sensor_msgs.msg.Image, self.front_image_cb)
        self.back_image_sub = rospy.Subscriber("~back_image", sensor_msgs.msg.Image, self.back_image_cb)
        self.debug_maker_pub = rospy.Publisher("bar_detection/debug_marker", MarkerArray, queue_size=10)
        self.max_marker_count = 2

        self.server = actionlib.SimpleActionServer("bar_detection/run_detector", LocalizeBarsAction, self.execute_action, False)
        self.server.start()

        project_pixel_to_ray_srv = "project_pixel_to_ray"
        rospy.loginfo("Waiting for service " + project_pixel_to_ray_srv)
        rospy.wait_for_service(project_pixel_to_ray_srv)
        self.project_pixel_to_ray = rospy.ServiceProxy(project_pixel_to_ray_srv,
                                                       hector_perception_msgs.srv.ProjectPixelTo3DRay)
        get_distance_to_obstacle_srv = "/move_group/get_distance_to_obstacle"
        rospy.loginfo("Waiting for service " + get_distance_to_obstacle_srv)
        rospy.wait_for_service(get_distance_to_obstacle_srv)
        self.get_distance_to_obstacle = rospy.ServiceProxy(get_distance_to_obstacle_srv,
                                                           hector_nav_msgs.srv.GetDistanceToObstacle)
        rospy.loginfo("Found all services.")

    def front_image_cb(self, image):
        self.last_front_image = image

    def back_image_cb(self, image):
        self.last_back_image = image

    def run_detection(self, detect_forward):
        last_image = self.last_front_image if detect_forward else self.last_back_image
        rospy.logdebug(" ### Starting detection")
        error = BarDetectionErrorType.NoError
        error_msg = ""
        bar_location = geometry_msgs.msg.PoseStamped()
        if last_image is not None:
            image_cv = self.bridge.imgmsg_to_cv2(last_image, desired_encoding="rgb8")
            detected_img, detections = self.detector.detect(image_cv)

            if len(detections) != 0:
                try:
                    first_bar_start_point, first_bar_center_point = self.estimate_global_points(detections[0])
                    second_bar_start_point, second_bar_center_point = self.estimate_global_points(detections[1])
                    if self.debug:
                        self.debug_add_marker([first_bar_start_point, first_bar_center_point, second_bar_start_point,
                                              second_bar_center_point])

                except BarDetectionError as e:
                    rospy.logerr(str(e))
                    return bar_location, error, error_msg

                rospy.loginfo("Successful detect bars!")

            detection_image_msg = self.bridge.cv2_to_imgmsg(detected_img, encoding="rgb8")
            self.detection_image_pub.publish(detection_image_msg)
            # TODO Calc position and orientation

        else:
            error_msg = "Detection skipped, because no image has been received yet."
            error = BarDetectionErrorType.NoImageError
            rospy.logwarn(error_msg)

        return bar_location, error, error_msg

    def estimate_global_points(self, detection):
        start_point_msg = geometry_msgs.msg.PointStamped()
        start_point_msg.point.x = detection.start[0]
        start_point_msg.point.y = detection.start[1]

        center_point_msg = geometry_msgs.msg.PointStamped()
        center_point_msg.point.x = detection.center[0]
        center_point_msg.point.y = detection.center[1]

        try:
            start_resp_project = self.project_pixel_to_ray(start_point_msg)
            center_resp_project = self.project_pixel_to_ray(center_point_msg)
        except rospy.ServiceException as e:
            raise BarDetectionError("ProjectPixelTo3DRay Service Exception",
                                    BarDetectionErrorType.ProjectPixelTo3DRayError)
        try:
            start_resp_raycast = self.get_distance_to_obstacle(start_resp_project.ray)
            center_resp_raycast = self.get_distance_to_obstacle(center_resp_project.ray)
        except rospy.ServiceException as e:
            raise BarDetectionError("GetDistanceToObstacle Service Exception",
                                    BarDetectionErrorType.GetDistanceToObstacleError)

        if start_resp_raycast.distance == -1 or center_resp_raycast.distance == -1:
            raise BarDetectionError("No intersection point found",
                                    BarDetectionErrorType.NoIntersectionPointError)

        return start_resp_raycast.end_point.point, center_resp_raycast.end_point.point

    def debug_add_marker(self, points):
        marker_array = MarkerArray()
        for n in range(len(points)):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.id = n
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            marker.pose.orientation.w = 1.0
            marker.pose.position.x = points[n].x
            marker.pose.position.y = points[n].y
            marker.pose.position.z = points[n].z

            marker_array.markers.append(marker)

        if self.debug:
            self.debug_maker_pub.publish(marker_array)

    def execute_action(self, goal):
        bar_localization, error, error_msg = self.run_detection(goal.front_cam)
        result = LocalizeBarsResult()
        result.success = error == BarDetectionErrorType.NoImageError
        result.error = error
        result.error_msg = error_msg
        result.pos = bar_localization

        self.server.set_succeeded(result)


if __name__ == "__main__":
    rospy.init_node("bar_detection")
    bar_detection = BarDetectionNode()
    rospy.spin()