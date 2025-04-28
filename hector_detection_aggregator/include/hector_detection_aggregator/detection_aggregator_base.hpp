//
// Created by finn on 4/24/25.
//

#ifndef DETECTION_AGGREGATOR_BASE_HPP
#define DETECTION_AGGREGATOR_BASE_HPP

#include <image_transport/image_transport.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <cv_bridge/cv_bridge.hpp>


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

/// Base class for different aggregation methods
class DetectionAggregatorBase
{
public:
  virtual ~DetectionAggregatorBase() = default;

  /// Called whenever a new camera image is received
  /// @return Indicates whether new aggregated data is available
  virtual bool AddImage(const sensor_msgs::msg::Image::ConstSharedPtr& image,
                        const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info) = 0;

  /// Called whenever a new detection message is received
  /// @return Indicates whether new aggregated data is available
  virtual bool AddDetection(const vision_msgs::msg::Detection2DArray::ConstSharedPtr& detections,
                            const rclcpp::MessageInfo& info) = 0;

  /// Called after either AddImage or AddDetection have indicated that new data is available
  /// @return A camera image and the detections that should be displayed on it
  virtual std::pair<cv_bridge::CvImageConstPtr, std::vector<DetectionEntry>> GetAggregatedData() = 0;

  /// Called regularly to give the aggregator information about currently active detectors
  virtual void UpdatePublishers(const std::vector<rclcpp::TopicEndpointInfo>& publisher_info) = 0;
};
}

#endif //DETECTION_AGGREGATOR_BASE_HPP
