#!/usr/bin/env python
from __future__ import print_function, division
import rospy
import cv2
import cv_bridge
import actionlib

import sensor_msgs.msg
import geometry_msgs.msg
import hector_perception_msgs.msg
import hector_perception_msgs.srv
import hector_worldmodel_msgs.msg
import hector_nav_msgs.srv

import barrels_detection


class BarrelsDetectionNode:
    def __init__(self):
        self.last_image = None
        self.bridge = cv_bridge.CvBridge()
        self.detector = barrels_detection.BarrelsDetection()

        self.image_projection_raycast_enabled = rospy.get_param("~image_projection_raycast", False)
        if self.image_projection_raycast_enabled:
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

        self.detection_image_pub = rospy.Publisher("~detection_image", sensor_msgs.msg.Image, queue_size=10, latch=True)
        self.perception_pub = rospy.Publisher("image_percept", hector_perception_msgs.msg.PerceptionDataArray,
                                              queue_size=10)
        self.world_model_pub = rospy.Publisher("/worldmodel/pose_percept", hector_worldmodel_msgs.msg.PosePercept,
                                               queue_size=10)
        self.image_sub = rospy.Subscriber("~image", sensor_msgs.msg.Image, self.image_cb)
        self.detect_as = actionlib.SimpleActionServer("/barrels_detection/run_detection",
                                                      hector_perception_msgs.msg.DetectObjectAction,
                                                      execute_cb=self.run_detection, auto_start=False)
        self.detect_as.start()

    def image_cb(self, image):
        self.last_image = image

    def run_detection(self, goal=None):
        rospy.logdebug(" ### Starting detection")
        if self.last_image is not None:
            image_cv = self.bridge.imgmsg_to_cv2(self.last_image, desired_encoding="rgb8")
            detections, detect_image = self.detector.detect(image_cv)
            detect_image = cv2.cvtColor(detect_image, cv2.COLOR_BGR2RGB)
            detection_image_msg = self.bridge.cv2_to_imgmsg(detect_image, encoding="bgr8")
            self.detection_image_pub.publish(detection_image_msg)
            self.publish_detections(detections)
            success = True
        else:
            rospy.logwarn("Barrels detection skipped, because no image has been received yet.")
            success = False

        if goal is not None:
            result = hector_perception_msgs.msg.DetectObjectResult()
            result.detection_success = success

            if success:
                self.detect_as.set_succeeded(result)
            else:
                self.detect_as.set_aborted(result, "Skipped, because has been image received.")

    def publish_detections(self, detections):
        array = hector_perception_msgs.msg.PerceptionDataArray()
        array.header.stamp = self.last_image.header.stamp
        array.perceptionType = "barrel"
        for detection in detections:
            perception_msg = hector_perception_msgs.msg.PerceptionData()
            perception_msg.percept_name = detection.name
            for point in detection.points:
                point_msg = geometry_msgs.msg.Point32()
                point_msg.x = point[0]
                point_msg.y = point[1]
                point_msg.z = 0
                perception_msg.polygon.points.append(point_msg)
            array.perceptionList.append(perception_msg)
        self.perception_pub.publish(array)

        if self.image_projection_raycast_enabled:
            for detection in detections:
                point_msg = geometry_msgs.msg.PointStamped()
                point_msg.point.x = detection.center[0]
                point_msg.point.y = detection.center[1]
                try:
                    resp_project = self.project_pixel_to_ray(point_msg)
                except rospy.ServiceException as e:
                    rospy.logerr("ProjectPixelTo3DRay Service Exception: " + str(e))
                    return

                try:
                    resp_raycast = self.get_distance_to_obstacle(resp_project.ray)
                except rospy.ServiceException as e:
                    rospy.logerr("GetDistanceToObstacle Service Exception: " + str(e))
                    return

                pose_msg = geometry_msgs.msg.PoseWithCovariance()

                pose_msg.pose.position = resp_raycast.end_point.point
                pose_msg.pose.orientation.w = 1

                pose_percept_msg = hector_worldmodel_msgs.msg.PosePercept()
                pose_percept_msg.pose = pose_msg
                pose_percept_msg.header.frame_id = "world"
                pose_percept_msg.header.stamp = self.last_image.header.stamp

                #pose_percept_msg.info.name = detection.name
                pose_percept_msg.info.class_id = detection.name
                pose_percept_msg.info.class_support = 1.0
                # pose_percept_msg.info.object_id = detection.name
                pose_percept_msg.info.object_support = 1.0

                self.world_model_pub.publish(pose_percept_msg)


if __name__ == "__main__":
    rospy.init_node("barrels_detection")
    hazmat_detection = BarrelsDetectionNode()

    hz = rospy.get_param("~detection_frequency", 0.2)
    rate = rospy.Rate(hz)
    while not rospy.is_shutdown():
        hazmat_detection.run_detection()
        try:
            rate.sleep()
        except rospy.exceptions.ROSTimeMovedBackwardsException as e:
            pass
