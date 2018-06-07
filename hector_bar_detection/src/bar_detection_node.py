#!/usr/bin/env python
from __future__ import print_function, division
import rospy
import cv_bridge

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

import sensor_msgs.msg
import geometry_msgs.msg
import hector_perception_msgs.msg
import hector_perception_msgs.srv
import hector_nav_msgs.srv

import bar_detection


class BarDetectionNode:
    def __init__(self):
        self.last_image = None
        self.bridge = cv_bridge.CvBridge()
        self.detector = bar_detection.BarDetection()
        self.detection_image_pub = rospy.Publisher("~detection_image", sensor_msgs.msg.Image, queue_size=10, latch=True)
        self.perception_pub = rospy.Publisher("image_percept", hector_perception_msgs.msg.PerceptionDataArray,
                                              queue_size=10)
        self.image_sub = rospy.Subscriber("~image", sensor_msgs.msg.Image, self.image_cb)
        self.debug_maker_pub = rospy.Publisher("bar_detection/debug_marker", MarkerArray, queue_size=10)
        self.marker_array = MarkerArray()
        self.max_marker_count = 2

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

    def image_cb(self, image):
        self.last_image = image

    def run_detection(self):
        rospy.logdebug(" ### Starting detection")
        if self.last_image is not None:
            image_cv = self.bridge.imgmsg_to_cv2(self.last_image, desired_encoding="rgb8")
            detected_img, detections = self.detector.detect(image_cv)

            for detection in detections:
                start_point_msg = geometry_msgs.msg.PointStamped()
                start_point_msg.point.x = detection.start[0]
                start_point_msg.point.y = detection.start[1]

                try:
                    start_resp_project = self.project_pixel_to_ray(start_point_msg)
                except rospy.ServiceException as e:
                    rospy.logerr("ProjectPixelTo3DRay Service Exception: " + str(e))
                    continue

                try:
                    start_resp_raycast = self.get_distance_to_obstacle(start_resp_project.ray)
                except rospy.ServiceException as e:
                    rospy.logerr("GetDistanceToObstacle Service Exception: " + str(e))
                    continue

                if start_resp_raycast.end_point.point.x == 0 and start_resp_raycast.end_point.point.y == 0 and start_resp_raycast.end_point.point.z == 0:
                    rospy.loginfo("No intersection point found")
                else:
                    self.debug_add_marker(start_resp_raycast.end_point.point)

            detection_image_msg = self.bridge.cv2_to_imgmsg(detected_img, encoding="bgr8")
            self.detection_image_pub.publish(detection_image_msg)
        else:
            rospy.logwarn("Detection skipped, because no image has been received yet.")

    def debug_add_marker(self, position):
        marker_id = len(self.marker_array.markers)
        if len(self.marker_array.markers) == self.max_marker_count:
            old_marker = self.marker_array.markers.pop(0)
            marker_id = old_marker.id

        rospy.logwarn("New ID: " + str(marker_id))

        marker = Marker()
        marker.header.frame_id = "world"
        marker.id = marker_id
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
        marker.pose.position.x = position.x
        marker.pose.position.y = position.y
        marker.pose.position.z = position.z

        self.marker_array.markers.append(marker)
        self.debug_maker_pub.publish(self.marker_array)


if __name__ == "__main__":
    rospy.init_node("bar_detection")
    bar_detection = BarDetectionNode()

    hz = rospy.get_param("~detection_frequency", 0.2)
    rate = rospy.Rate(hz)
    while not rospy.is_shutdown():
        bar_detection.run_detection()
        try:
            rate.sleep()
        except rospy.exceptions.ROSTimeMovedBackwardsException as e:
            pass
