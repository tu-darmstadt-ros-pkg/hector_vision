#ifndef HECTOR_DETECTION_AGGREGATOR_H_
#define HECTOR_DETECTION_AGGREGATOR_H_

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <opencv/cv.h>
#include <cv_bridge/cv_bridge.h>

#include <hector_worldmodel_msgs/ImagePercept.h>
#include <hector_perception_msgs/PerceptionDataArray.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/image_encodings.h>

#include <dynamic_reconfigure/server.h>

#include <hector_detection_aggregator/HectorDetectionAggregatorConfig.h>

namespace hector_detection_aggregator
{
class DetectionAggregator{
public:
    DetectionAggregator();
    ~DetectionAggregator();
    void createImage();

private:
    void updateDetections();
    void imageCallback(const sensor_msgs::ImageConstPtr& img);
    void imageDetectionCallback(const hector_perception_msgs::PerceptionDataArrayConstPtr& percept);
    void dynRecParamCallback(HectorDetectionAggregatorConfig &config, uint32_t level);

    image_transport::CameraPublisher image_detected_pub_;

    ros::Subscriber image_percept_sub_;
    image_transport::Subscriber image_sub_;
    image_transport::CameraSubscriber camera_sub_;

    dynamic_reconfigure::Server<HectorDetectionAggregatorConfig> dyn_rec_server_;

    cv_bridge::CvImageConstPtr img_current_grey_ptr_;
    cv_bridge::CvImageConstPtr img_current_col_ptr_;

    std::map<std::string, hector_perception_msgs::PerceptionDataArray> detection_map_;
    std::map<std::string, cv::Scalar> color_map_;

     //params
    ros::Duration storage_duration_;

};
}
#endif
