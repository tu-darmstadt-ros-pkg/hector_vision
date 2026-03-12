#ifndef HECTOR_DETECTION_AGGREGATOR_NODE_HPP_
#define HECTOR_DETECTION_AGGREGATOR_NODE_HPP_

#include "detection_aggregator_base.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <std_msgs/msg/bool.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

typedef vision_msgs::msg::Detection2DArray Detection2DArray;

namespace hector_detection_aggregator
{
class DetectionAggregatorNode
{
public:
  explicit DetectionAggregatorNode( const rclcpp::Node::SharedPtr &node );
  DetectionAggregatorNode( DetectionAggregatorNode &da ) = delete;
  DetectionAggregatorNode( DetectionAggregatorNode &&da ) = delete;
  ~DetectionAggregatorNode();
  void createImage();

private:
  bool enabled_;
  bool has_subscribers_;
  std::string robot_namespace_;
  /// Detection topic chosen by parameter
  std::string detection_topic_;
  /// Detection topic overridden by remapping
  std::string real_detection_topic_;
  // rclcpp::Duration storage_duration_;

  std::shared_ptr<DetectionAggregatorBase> detection_aggregator_;

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<rclcpp::ParameterEventHandler> param_subscriber_;
  rclcpp::ParameterCallbackHandle::SharedPtr enabled_callback_handle_;
  rclcpp::ParameterCallbackHandle::SharedPtr storage_duration_callback_handle_;
  rclcpp::TimerBase::SharedPtr check_environment_timer_;
  std::shared_ptr<image_transport::ImageTransport> image_transport_;

  // std::map<std::pair<std::string, std::string>, vision_msgs::msg::Detection2D> detection_map_;
  std::map<std::string, cv::Scalar> color_map_;

  image_transport::Publisher image_detected_pub_;

  rclcpp::Subscription<Detection2DArray>::SharedPtr image_percept_sub_;
  image_transport::Subscriber image_subscriber_;

  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enabled_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr enabled_pub_;

  void updateDetections();
  void imageCallback( const sensor_msgs::msg::Image::ConstSharedPtr &image );
  void imageDetectionCallback( const Detection2DArray::ConstSharedPtr &percept,
                               const rclcpp::MessageInfo &info );

  void publishEnableStatus() const;
  void enabledCallback( const bool &enabled );
  void msgEnabledCallback( const std_msgs::msg::Bool::ConstSharedPtr &enabled ) const;

  void startSubscribers();
  void stopSubscribers();
  void checkPublisherSubscriptions();
  void checkEnvironmentCallback();
};
} // namespace hector_detection_aggregator
#endif
