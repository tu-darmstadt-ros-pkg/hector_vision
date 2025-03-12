#ifndef HECTOR_DETECTION_AGGREGATOR_HPP_
#define HECTOR_DETECTION_AGGREGATOR_HPP_

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp/rclcpp.hpp>
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
  void updateDetections();
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& img);
  void imageDetectionCallback(const Detection2DArray::ConstSharedPtr& percept);
  // TODO Parameter update
  // void dynRecParamCallback(HectorDetectionAggregatorConfig &config, uint32_t level);

  void connectCb();
  void startSubscribers();
  void stopSubscribers();

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<rclcpp::ParameterEventHandler> param_subscriber_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> storage_duration_callback_handle_;
  std::shared_ptr<image_transport::ImageTransport> image_transport_;

  image_transport::CameraPublisher image_detected_pub_;

  rclcpp::Subscription<Detection2DArray>::SharedPtr image_percept_sub_;
  image_transport::Subscriber image_sub_;
  image_transport::CameraSubscriber camera_sub_;

  cv_bridge::CvImageConstPtr img_current_grey_ptr_;
  cv_bridge::CvImageConstPtr img_current_col_ptr_;

  std::map<std::pair<std::string, std::string>, vision_msgs::msg::Detection2D> detection_map_;
  std::map<std::string, cv::Scalar> color_map_;

   //params
  rclcpp::Duration storage_duration_;
};
}
#endif
