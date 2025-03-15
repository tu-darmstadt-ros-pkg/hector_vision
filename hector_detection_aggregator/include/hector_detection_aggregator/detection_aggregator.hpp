#ifndef HECTOR_DETECTION_AGGREGATOR_HPP_
#define HECTOR_DETECTION_AGGREGATOR_HPP_

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

typedef vision_msgs::msg::Detection2DArray Detection2DArray;

namespace hector_detection_aggregator
{
class DetectionAggregator
{
public:
  explicit DetectionAggregator(const rclcpp::Node::SharedPtr& node);
  DetectionAggregator(DetectionAggregator& da) = delete;
  DetectionAggregator(DetectionAggregator&& da) = delete;
  ~DetectionAggregator();
  void createImage();

private:
  bool enabled_;
  bool has_subscribers_;
  rclcpp::Duration storage_duration_;

  cv_bridge::CvImageConstPtr current_color_image_;
  // cv_bridge::CvImageConstPtr current_grey_image_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr current_camera_info_;

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<rclcpp::ParameterEventHandler> param_subscriber_;
  rclcpp::ParameterCallbackHandle::SharedPtr enabled_callback_handle_;
  rclcpp::ParameterCallbackHandle::SharedPtr storage_duration_callback_handle_;
  rclcpp::TimerBase::SharedPtr check_subscribers_timer_;
  std::shared_ptr<image_transport::ImageTransport> image_transport_;

  std::map<std::pair<std::string, std::string>, vision_msgs::msg::Detection2D> detection_map_;
  std::map<std::string, cv::Scalar> color_map_;

  image_transport::CameraPublisher image_detected_pub_;

  rclcpp::Subscription<Detection2DArray>::SharedPtr image_percept_sub_;
  image_transport::CameraSubscriber camera_subscriber_;

  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enabled_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr enabled_pub_;

  void updateDetections();
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& image,
                     const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info);
  void imageDetectionCallback(const Detection2DArray::ConstSharedPtr& percept);

  void publishEnableStatus() const;
  void enabledCallback(const bool& enabled);
  void msgEnabledCallback(const std_msgs::msg::Bool::ConstSharedPtr& enabled) const;

  void startSubscribers();
  void stopSubscribers();
  void publisherSubscriptionCallback();
};
}
#endif
