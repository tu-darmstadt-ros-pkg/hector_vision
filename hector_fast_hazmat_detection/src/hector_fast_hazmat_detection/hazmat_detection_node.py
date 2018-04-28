#!/usr/bin/env python
from __future__ import print_function, division
import rospy
import numpy as np
import cv2
import cv_bridge
import os
import pickle
import actionlib

import sensor_msgs.msg
import std_msgs.msg
import geometry_msgs.msg
import hector_perception_msgs.msg
import hector_perception_msgs.srv
import hector_worldmodel_msgs.msg
import hector_nav_msgs.srv

import hazmat_detection

DEBUG = True
DEBUG_FOLDER = None  # "/home/username/debug_hazmat"


class HazmatDetectionNode:
    def __init__(self, template_folder, verbose):
        self.last_image = None
        self.last_stamp = None
        self.bridge = cv_bridge.CvBridge()
        self.detector = hazmat_detection.HazmatSignDetector(template_folder)

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

        self.enabled_sub = rospy.Subscriber("~enabled", std_msgs.msg.Bool, self.enabled_cb)
        self.enabled_status_pub = rospy.Publisher("~enabled_status", std_msgs.msg.Bool, queue_size=10, latch=True)
        self.enabled = rospy.get_param("~periodic_detection", False)
        self.publish_enabled_status()

        self.debug_image_pub = rospy.Publisher("~debug_image", sensor_msgs.msg.Image, queue_size=10, latch=True)
        self.perception_pub = rospy.Publisher("image_percept", hector_perception_msgs.msg.PerceptionDataArray,
                                              queue_size=10)
        self.world_model_pub = rospy.Publisher("/worldmodel/pose_percept", hector_worldmodel_msgs.msg.PosePercept,
                                               queue_size=10)
        self.image_sub = rospy.Subscriber("~image", sensor_msgs.msg.Image, self.image_cb)
        self.detect_as = actionlib.SimpleActionServer("/hazmat_detection/run_detection",
                                                      hector_perception_msgs.msg.DetectObjectAction,
                                                      execute_cb=self.run_detection, auto_start=False)
        self.detect_as.start()
        if DEBUG and DEBUG_FOLDER is not None and not os.path.isdir(DEBUG_FOLDER):
            os.mkdir(DEBUG_FOLDER)

    def image_cb(self, image):
        self.last_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="rgb8")
        # cv_img = cv2.cvtColor(self.last_image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("hazmat_wall.png", cv_img)
        self.last_stamp = image.header.stamp

    def publish_enabled_status(self):
        bool_msg = std_msgs.msg.Bool()
        bool_msg.data = self.enabled
        self.enabled_status_pub.publish(bool_msg)

    def enabled_cb(self, enabled_msg):
        self.enabled = enabled_msg.data
        status = "Enabled" if self.enabled else "Disabled"
        rospy.loginfo(status + " periodic hazmat detection.")
        self.publish_enabled_status()

    def run_detection(self, goal=None):
        rospy.logdebug(" ### Starting detection")
        if self.last_image is not None:
            image = self.last_image
            result = self.detector.detect(image, debug=DEBUG and DEBUG_FOLDER is not None)
            if DEBUG:
                if DEBUG_FOLDER is not None:
                    cv2.imwrite(os.path.join(DEBUG_FOLDER, str(self.last_stamp) + ".png"),
                                cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    with open(os.path.join(DEBUG_FOLDER, str(self.last_stamp)+".pickle"), 'wb') as handle:
                        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
                debug_image = image
                for detection in result.detections:
                    cv2.drawContours(debug_image, [detection.contour], 0, np.array([255, 0, 0]), 2)
                    (x, y, w, h) = cv2.boundingRect(detection.contour)
                    cv2.putText(debug_image, detection.name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, np.array([255, 0, 0]), 2)
                debug_image_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="rgb8")
                if DEBUG_FOLDER is not None:
                    cv2.imwrite(os.path.join(DEBUG_FOLDER, str(self.last_stamp) + "_result.png"),
                                cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
                self.debug_image_pub.publish(debug_image_msg)
            self.publish_detections(result.detections)
            success = True
        else:
            rospy.logwarn("Hazmat detection skipped, because no image has been received yet.")
            success = False

        if goal is not None:
            result = hector_perception_msgs.msg.DetectObjectResult()
            result.detection_success = success

            if success:
                self.detect_as.set_succeeded(result)
            else:
                self.detect_as.set_aborted(result, "Skipped, because no image was received.")

    def publish_detections(self, detections):
        array = hector_perception_msgs.msg.PerceptionDataArray()
        array.header.stamp = self.last_stamp
        array.perceptionType = "hazmat"
        for detection in detections:
            rospy.loginfo("Detected hazmat: " + detection.name)
            perception_msg = hector_perception_msgs.msg.PerceptionData()
            perception_msg.percept_name = detection.name
            for point in detection.contour:
                point_msg = geometry_msgs.msg.Point32()
                point_msg.x = point[0][0]
                point_msg.y = point[0][1]
                point_msg.z = 0
                perception_msg.polygon.points.append(point_msg)
            array.perceptionList.append(perception_msg)
        self.perception_pub.publish(array)

        if self.image_projection_raycast_enabled:
            for detection in detections:
                point_msg = geometry_msgs.msg.PointStamped()
                point_msg.point.x = detection.contour[0][0][0]
                point_msg.point.y = detection.contour[0][0][1]
                try:
                    resp_project = self.project_pixel_to_ray(point_msg)
                except rospy.ServiceException as e:
                    rospy.logerr("ProjectPixelTo3DRay Service Exception: " + str(e))
                    continue

                try:
                    resp_raycast = self.get_distance_to_obstacle(resp_project.ray)
                except rospy.ServiceException as e:
                    rospy.logerr("GetDistanceToObstacle Service Exception: " + str(e))
                    continue

                if resp_raycast.end_point.point.x == 0 and resp_raycast.end_point.point.y == 0 and resp_raycast.end_point.point.z == 0:
                    continue
                    
                pose_msg = geometry_msgs.msg.PoseWithCovariance()
                pose_msg.pose.position = resp_raycast.end_point.point
                pose_msg.pose.orientation.w = 1

                pose_percept_msg = hector_worldmodel_msgs.msg.PosePercept()
                pose_percept_msg.pose = pose_msg
                pose_percept_msg.header.frame_id = "world"
                pose_percept_msg.header.stamp = self.last_stamp

                #pose_percept_msg.info.name = detection.name
                pose_percept_msg.info.class_id = detection.name
                pose_percept_msg.info.class_support = 1.0
                # pose_percept_msg.info.object_id = detection.name
                pose_percept_msg.info.object_support = 1.0

                self.world_model_pub.publish(pose_percept_msg)


if __name__ == "__main__":
    rospy.init_node("hazmat_detection_node")
    hazmat_detection = HazmatDetectionNode(rospy.get_param("~model_folder"), verbose=False)
    #hazmat_detection.sift_matcher.min_match_count = rospy.get_param("~min_match_count", 5)
    #hazmat_detection.sift_matcher.clahe = None  # disable clahe
    ## hazmat_detection.load_models("~models")
    #try:
    #    hazmat_detection.sift_matcher.load_models_from_folder(rospy.get_param("~model_folder"))
    #except KeyError as e:
    #    rospy.logerr("Model folder not set: " + str(e))

    # hazmat_detection.sift_matcher.set_clahe(2, (8, 8))

    hz = rospy.get_param("~detection_frequency", 0.2)
    rate = rospy.Rate(hz)
    while not rospy.is_shutdown():
        if hazmat_detection.enabled:
            hazmat_detection.run_detection()
        try:
            rate.sleep()
        except rospy.exceptions.ROSTimeMovedBackwardsException as e:
            pass
