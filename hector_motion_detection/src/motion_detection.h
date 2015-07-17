#ifndef _HECTOR_MOTION_DETECTION_H_
#define _HECTOR_MOTION_DETECTION_H_

#include <ros/ros.h>

#include <opencv/cv.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <image_transport/image_transport.h>

#include <std_msgs/String.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CompressedImage.h>

#include <hector_worldmodel_msgs/ImagePercept.h>

#include <dynamic_reconfigure/server.h>
#include <hector_motion_detection/MotionDetectionConfig.h>

using hector_motion_detection::MotionDetectionConfig;

class MotionDetection{
public:
    MotionDetection();
    ~MotionDetection();
private:
    void imageCallback(const sensor_msgs::ImageConstPtr& img);
    //void mappingCallback(const thermaleye_msgs::Mapping& mapping);
    void dynRecParamCallback(MotionDetectionConfig &config, uint32_t level);

    image_transport::Subscriber image_sub_;

    ros::Publisher image_percept_pub_;
    image_transport::CameraSubscriber camera_sub_;
    image_transport::CameraPublisher image_motion_pub_;
    image_transport::CameraPublisher image_detected_pub_;

    dynamic_reconfigure::Server<MotionDetectionConfig> dyn_rec_server_;

    cv_bridge::CvImageConstPtr img_prev_ptr_;
    cv_bridge::CvImageConstPtr img_current_ptr_;
    cv_bridge::CvImageConstPtr img_current_col_ptr_;

    int motion_detect_threshold_;
    double min_percept_size, max_percept_size;
    double min_density;
    std::string percept_class_id_;

};

#endif
