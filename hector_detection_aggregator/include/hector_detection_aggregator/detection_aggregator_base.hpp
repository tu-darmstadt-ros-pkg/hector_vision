//
// Created by finn on 4/24/25.
//

#ifndef DETECTION_AGGREGATOR_BASE_HPP
#define DETECTION_AGGREGATOR_BASE_HPP

#include <image_transport/image_transport.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>


namespace hector_detection_aggregator
{
struct DetectionEntry
{
  std::string type;
  std::string id;
  vision_msgs::msg::Detection2D detection;

  DetectionEntry(const std::string& type, const std::string& id, const vision_msgs::msg::Detection2D& detection)
    : type(type), id(id), detection(detection) {}
};

class DetectionAggregatorBase
{
public:
  virtual ~DetectionAggregatorBase() = default;

  virtual bool AddImage(const sensor_msgs::msg::Image::ConstSharedPtr& image,
                const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info) = 0;

  virtual bool AddDetection(const vision_msgs::msg::Detection2DArray::ConstSharedPtr& detections,
                    const rclcpp::MessageInfo& info) = 0;

  virtual std::pair<cv_bridge::CvImageConstPtr, std::vector<DetectionEntry>> GetAggregatedData() = 0;

  virtual void UpdatePublishers(const std::vector<rclcpp::TopicEndpointInfo>& publisher_info) = 0;
};
}

#endif //DETECTION_AGGREGATOR_BASE_HPP
