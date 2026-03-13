#ifndef HECTOR_DETECTION_AGGREGATOR_HPP_
#define HECTOR_DETECTION_AGGREGATOR_HPP_

#include "detection_aggregator_base.hpp"

#include <hector_ros2_utils/node.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>

#include <image_transport/image_transport.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

typedef vision_msgs::msg::Detection2DArray Detection2DArray;

namespace hector_detection_aggregator
{
class DetectionAggregator : public hector::Node
{
public:
  DetectionAggregator( const rclcpp::NodeOptions &options );
  // DetectionAggregator( DetectionAggregator &da ) = delete;
  // DetectionAggregator( DetectionAggregator &&da ) = delete;

  void createImage();

private:
  bool has_subscribers_;
  std::string robot_namespace_;
  /// Detection topic chosen by parameter
  std::string detection_topic_;
  /// Detection topic overridden by remapping
  std::string real_detection_topic_;
  // rclcpp::Duration storage_duration_;

  std::shared_ptr<DetectionAggregatorBase> detection_aggregator_;

  std::shared_ptr<rclcpp::ParameterEventHandler> param_subscriber_;
  rclcpp::ParameterCallbackHandle::SharedPtr storage_duration_callback_handle_;
  rclcpp::TimerBase::SharedPtr start_image_transfer_timer_;
  rclcpp::TimerBase::SharedPtr check_environment_timer_;
  std::shared_ptr<image_transport::ImageTransport> image_transport_;

  // std::map<std::pair<std::string, std::string>, vision_msgs::msg::Detection2D> detection_map_;
  std::map<std::string, cv::Scalar> color_map_;

  image_transport::Publisher image_detected_pub_;

  rclcpp::Subscription<Detection2DArray>::SharedPtr image_percept_sub_;
  image_transport::Subscriber image_subscriber_;

  void startImageTransferCallback();
  void imageCallback( const sensor_msgs::msg::Image::ConstSharedPtr &image );
  void imageDetectionCallback( const Detection2DArray::ConstSharedPtr &percept,
                               const rclcpp::MessageInfo &info );

  void startSubscribers();
  void stopSubscribers();
  void checkPublisherSubscriptions();
  void checkEnvironmentCallback();
};
} // namespace hector_detection_aggregator
#endif
