#ifndef _HECTOR_MOTION_DETECTION_H_
#define _HECTOR_MOTION_DETECTION_H_

#include <ros/ros.h>
#include <ros/callback_queue.h>

#include <image_transport/image_transport.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/image_encodings.h>
#include <hector_worldmodel_msgs/ImagePercept.h>
#include <hector_perception_msgs/PerceptionDataArray.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <dynamic_reconfigure/server.h>
#include <hector_motion_detection/MotionDetectionConfig.h>

using hector_motion_detection::MotionDetectionConfig;

namespace hector_motion_detection {

class MotionDetection{
public:
    MotionDetection(ros::NodeHandle& nh, ros::NodeHandle& pnh);
    void publishEnableStatus();
private:
    void imageCallback(const sensor_msgs::ImageConstPtr& img); //, const sensor_msgs::CameraInfoConstPtr& info);
    void enabledCallback(const std_msgs::BoolConstPtr& enabled);
    void dynRecParamCallback(MotionDetectionConfig &config, uint32_t level);

    void connectCb();
    void startSubscribers();
    void shutdownSubscribers();

    boost::mutex connect_mutex_;

    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;

    bool enabled_;

    ros::Publisher image_percept_pub_;
    ros::Publisher image_perception_pub;
    image_transport::Subscriber image_sub_;
    ros::Subscriber enabled_sub_;
    ros::Publisher enabled_pub_;

    image_transport::CameraSubscriber camera_sub_;
    image_transport::CameraPublisher image_motion_pub_;
    image_transport::CameraPublisher image_detected_pub_;

    dynamic_reconfigure::Server<MotionDetectionConfig> dyn_rec_server_;
    dynamic_reconfigure::Server<MotionDetectionConfig>::CallbackType dyn_rec_type_;

    bool first_image_received_;
    cv::Mat accumulated_image_;



    cv::Ptr<cv::BackgroundSubtractorMOG2> bg_subtractor_; //with regards to shadows
    int detectionLimit_; //the maximal number of detections to make/objects to track
    int min_area_; //to filter smaller areas
    int max_area_; //to filter bigger areas
    double learning_rate_; // learning rate of bg subtractor
    bool automatic_learning_rate_; // set learning rate automatically
    int erosion_iterations_, dilation_iterations_; //for controlling the iterations of erosion/deliation
    bool shadows_; //control if shadows should be tracked
    bool debug_contours_;
    double moving_average_weight_;
    int activation_threshold_;

    image_transport::CameraPublisher image_background_subtracted_pub_; //for publishing subtracted image


};

}

#endif
