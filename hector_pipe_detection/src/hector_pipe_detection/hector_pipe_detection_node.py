#!/usr/bin/env python
import rospy
import sensor_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
import image_geometry
import tf2_ros

import cv2
import cv_bridge
import numpy as np


def low_pass_filter(old, new, factor):
    return (1 - factor) * old + factor * new


class PipeDetectionNode:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cv_bridge = cv_bridge.CvBridge()
        self.circle_pos = None

        self.image_sub = rospy.Subscriber("image", sensor_msgs.msg.Image, self.image_cb)
        self.camera_info_sub = rospy.Subscriber("camera_info", sensor_msgs.msg.CameraInfo, self.camera_info_cb)
        self.camera_info = None

        self.enabled_status_pub = rospy.Publisher("~enabled_status", std_msgs.msg.Bool, queue_size=100, latch=True)
        self.enabled_sub = rospy.Subscriber("~enabled", std_msgs.msg.Bool, self.enabled_cb)
        self._enabled = False
        self.enabled = rospy.get_param("enabled", False)
        self.publish_enabled_status()

        self.circle_image_pub = rospy.Publisher("~detected_circle", sensor_msgs.msg.Image, queue_size=100, latch=False)
        self.detected_pose_pub = rospy.Publisher("~detected_pose", geometry_msgs.msg.PoseStamped, queue_size=100, latch=False)

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value
        self.publish_enabled_status()

    def image_cb(self, image_msg):
        if self.enabled:
            self.detect_pipe(image_msg)

    def camera_info_cb(self, camera_info_msg):
        self.camera_info = camera_info_msg

    def enabled_cb(self, bool_msg):
        self.enabled = bool_msg.data
        self.publish_enabled_status()

    def publish_enabled_status(self):
        status = "Enabled" if self.enabled else "Disabled"
        rospy.loginfo(status + " pipe detection.")
        self.enabled_status_pub.publish(std_msgs.msg.Bool(self._enabled))

    def detect_pipe(self, image):
        cv_image = self.cv_bridge.imgmsg_to_cv2(image)
        # cv2.imwriteimage
        image_gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.blur(image_gray, (15, 15))
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 1000, param1=50, param2=30, minRadius=100,
                                   maxRadius=0)
        if circles is None:
            return False

        circles = np.array(circles[0])
        if self.circle_pos is None:
            self.circle_pos = circles[0]
        else:
            self.circle_pos = low_pass_filter(self.circle_pos, circles[0], 0.15)

        circle_image = cv_image.copy()
        cv2.circle(circle_image, (self.circle_pos[0], self.circle_pos[1]), self.circle_pos[2], (0, 255, 0), 2)  # draw the outer circle
        cv2.circle(circle_image, (self.circle_pos[0], self.circle_pos[1]), 2, (0, 0, 255), 3)  # draw the center of the circle
        self.circle_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(circle_image, encoding="rgb8"))

        if self.camera_info is None:
            rospy.logwarn_throttle(1, "No camera info received. Can't inverse pixel.")
            return False

        model = image_geometry.PinholeCameraModel()
        model.fromCameraInfo(self.camera_info)
        ray = model.projectPixelTo3dRay(self.circle_pos[0:2])
        # TODO set length with known radius of circle
        self.publish_ray(image, ray)
        return True

    def publish_ray(self, image, ray):
        pose_msg = geometry_msgs.msg.PoseStamped()
        pose_msg.header.stamp = image.header.stamp
        pose_msg.header.frame_id = image.header.frame_id
        pose_msg.pose.position.x = ray[0]
        pose_msg.pose.position.y = ray[1]
        pose_msg.pose.position.z = ray[2]

        try:
            transform = self.tf_buffer.lookup_transform("base_link", image.header.frame_id, image.header.stamp)
        except Exception as e:
            rospy.logwarn("TF Exception: " + str(e))
        else:
            pose_msg.pose.orientation = transform.transform.rotation
        self.detected_pose_pub.publish(pose_msg)


if __name__ == "__main__":
    rospy.init_node("pipe_detection_node")
    pipe_detection_node = PipeDetectionNode()
    rospy.spin()
