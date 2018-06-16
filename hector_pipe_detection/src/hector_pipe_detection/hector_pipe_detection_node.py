#!/usr/bin/env python
from __future__ import division, print_function
import rospy
import sensor_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
import image_geometry
import tf2_ros

import cv2
import cv_bridge
import numpy as np

from circle_detection import find_outer_circle, DebugInfo

DEBUG = True

def low_pass_filter(old, new, factor):
    return (1 - factor) * old + factor * new


class PipeDetectionNode:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cv_bridge = cv_bridge.CvBridge()
        self.circle_pos = None
        self.circle_radius = rospy.get_param("circle_radius", 0.025)

        self.camera_info = None
        self.camera_info_sub = rospy.Subscriber("camera_info", sensor_msgs.msg.CameraInfo, self.camera_info_cb)

        self.model = None
        self.meter_to_pixel_ratio = None
        if self.wait_for_camera_info(5):
            self.model = image_geometry.PinholeCameraModel()
            self.model.fromCameraInfo(self.camera_info)
            self.calc_m_to_pixel_ratio(self.model)

        self.last_image = None
        self.image_sub = rospy.Subscriber("image", sensor_msgs.msg.Image, self.image_cb)

        self.enabled_status_pub = rospy.Publisher("~enabled_status", std_msgs.msg.Bool, queue_size=100, latch=True)
        self.enabled_sub = rospy.Subscriber("~enabled", std_msgs.msg.Bool, self.enabled_cb)
        self._enabled = False
        self.enabled = rospy.get_param("~enabled", False)
        self.publish_enabled_status()

        # Debug publishers
        if DEBUG:
            self.edge_image_pub = rospy.Publisher("~edge_image", sensor_msgs.msg.Image, queue_size=1, latch=True)
            self.contours_image_pub = rospy.Publisher("~contours", sensor_msgs.msg.Image, queue_size=1, latch=True)
            self.filtered_contours_image_pub = rospy.Publisher("~filtered_contours", sensor_msgs.msg.Image, queue_size=1, latch=True)
            self.sub_edge_image_pub = rospy.Publisher("~sub_image_edges", sensor_msgs.msg.Image, queue_size=1, latch=True)
            self.sub_contours_image_pub = rospy.Publisher("~sub_image_contours", sensor_msgs.msg.Image, queue_size=1, latch=True)
            self.detection_image_pub = rospy.Publisher("~detection_image", sensor_msgs.msg.Image, queue_size=1, latch=True)
        self.detected_pose_pub = rospy.Publisher("~detected_pose", geometry_msgs.msg.PoseStamped, queue_size=1, latch=False)

    def wait_for_camera_info(self, timeout):
        rospy.loginfo("Waiting for camera_info")
        start = rospy.Time.now()
        rate = rospy.Rate(1)
        while self.camera_info is None:
            now = rospy.Time.now()
            if (now - start).to_sec() > timeout:
                rospy.logwarn("Timed out waiting for camera_info")
                return False
            else:
                rate.sleep()
        return True

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value
        self.publish_enabled_status()

    def image_cb(self, image_msg):
        self.last_image = image_msg

    def camera_info_cb(self, camera_info_msg):
        self.camera_info = camera_info_msg

    def enabled_cb(self, bool_msg):
        self.enabled = bool_msg.data
        self.publish_enabled_status()

    def publish_enabled_status(self):
        status = "Enabled" if self.enabled else "Disabled"
        rospy.loginfo(status + " pipe detection.")
        self.enabled_status_pub.publish(std_msgs.msg.Bool(self._enabled))

    def draw_circle(self, image, circle):
        circle_image = image.copy()
        cv2.circle(circle_image, (int(circle[0]), int(circle[1])), int(circle[2]), (255, 0, 0), 4)  # draw circle
        cv2.circle(circle_image, (int(circle[0]), int(circle[1])), 2, (0, 0, 255), 6)  # draw circle center
        return circle_image

    def detect_pipe(self, image):
        if image is None:
            return False
        cv_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="rgb8")
        #cv2.imwrite("pipe_image.png", cv_image)
        rospy.loginfo("Received image")
        debug_info = None
        if DEBUG:
            debug_info = DebugInfo()
        center, radius = find_outer_circle(cv_image, debug_info)
        if DEBUG:
            self.edge_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug_info.edge_image, encoding="8UC1"))
            self.contours_image_pub.publish(
                self.cv_bridge.cv2_to_imgmsg(debug_info.get_contours_image(), encoding="rgb8"))
            self.filtered_contours_image_pub.publish(
                self.cv_bridge.cv2_to_imgmsg(debug_info.get_filtered_contours_image(), encoding="rgb8"))
            if debug_info.sub_edge_image is not None:
                self.sub_edge_image_pub.publish(
                    self.cv_bridge.cv2_to_imgmsg(debug_info.sub_edge_image, encoding="8UC1"))
                self.sub_contours_image_pub.publish(
                    self.cv_bridge.cv2_to_imgmsg(debug_info.get_sub_contours_image(), encoding="rgb8"))
        if center is None:
            return False
        circle = np.array([int(center[0]), int(center[1]), int(radius)])

        if self.circle_pos is None:
            self.circle_pos = circle
        else:
            self.circle_pos = low_pass_filter(self.circle_pos, circle, 0.9)  # TODO: Is this really necessary?

        if DEBUG:
            detection_image = self.draw_circle(cv_image, self.circle_pos)
            self.detection_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(detection_image, encoding="rgb8"))

        if self.model is None:
            rospy.logwarn("No camera info received. Can't use model.")
            return False

        distance = self.meter_to_pixel(self.circle_radius) / self.circle_pos[2]
        print("distance: ", distance)
        ray = np.array(self.model.projectPixelTo3dRay(self.circle_pos[0:2]))
        point = distance * ray / np.linalg.norm(ray)

        self.publish_pose(image, point)
        return True

    def calc_m_to_pixel_ratio(self, model):
        """

        :type model: image_geometry.PinholeCameraModel
        """
        x1 = np.array([0.5, 0, 1])
        x2 = np.array([-0.5, 0, 1])

        p1 = np.array(model.project3dToPixel(x1))
        p2 = np.array(model.project3dToPixel(x2))

        self.meter_to_pixel_ratio = np.linalg.norm(p1 - p2) # Or simply camera_info.K[0, 0]

    def meter_to_pixel(self, meter):
        if self.meter_to_pixel_ratio is not None:
            return self.meter_to_pixel_ratio * meter
        else:
            rospy.logwarn("meter_to_pixel_ratio hasn't been calculated yes")
            return 0

    def pixel_to_meter(self, pixel):
        if self.meter_to_pixel_ratio is not None:
            return pixel / self.meter_to_pixel_ratio
        else:
            rospy.logwarn("meter_to_pixel_ratio hasn't been calculated yes")
            return 0

    def publish_pose(self, image, ray):
        pose_msg = geometry_msgs.msg.PoseStamped()
        pose_msg.header.stamp = image.header.stamp
        pose_msg.header.frame_id = image.header.frame_id
        pose_msg.pose.position.x = ray[0]
        pose_msg.pose.position.y = ray[1]
        pose_msg.pose.position.z = ray[2]

        try:
            transform = self.tf_buffer.lookup_transform("base_link", image.header.frame_id, image.header.stamp)
        except Exception as e:
            rospy.logwarn_throttle(5, "TF Exception: " + str(e))
        else:
            pose_msg.pose.orientation = transform.transform.rotation
        self.detected_pose_pub.publish(pose_msg)


if __name__ == "__main__":
    rospy.init_node("pipe_detection_node")
    pipe_detection_node = PipeDetectionNode()

    hz = rospy.get_param("~detection_frequency", 0.5)
    rate = rospy.Rate(hz)
    while not rospy.is_shutdown():
        pipe_detection_node.detect_pipe(pipe_detection_node.last_image)
        try:
            rate.sleep()
        except rospy.exceptions.ROSTimeMovedBackwardsException as e:
            pass
