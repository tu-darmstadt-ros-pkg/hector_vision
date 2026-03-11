#ifndef HECTOR_MOTION_DETECTION_HPP
#define HECTOR_MOTION_DETECTION_HPP

#include <hector_ros2_utils/node.hpp>
#include <rclcpp/rclcpp.hpp>

#include <image_transport/image_transport.hpp>
#include <std_msgs/msg/bool.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

namespace hector_motion_detection
{

class MotionDetection : public hector::Node
{
public:
  MotionDetection( const rclcpp::NodeOptions &options );
  void publishEnableStatus() const;

private:
  void imageCallback( const sensor_msgs::msg::Image::ConstSharedPtr
                          &img ); //, const sensor_msgs::CameraInfoConstPtr& info);
  void enabledCallback( const std_msgs::msg::Bool::ConstSharedPtr &enabled );
  void publisherSubscriptionCallback();

  rclcpp::TimerBase::SharedPtr check_subscriptions_timer_;

  void startSubscribers();
  void stopSubscribers();

  image_transport::ImageTransport it_;

  bool enabled_;
  bool has_subscribers_;

  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enabled_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr enabled_pub_;

  image_transport::Subscriber image_sub_;

  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr image_perception_pub_;

  image_transport::CameraPublisher image_motion_pub_;
  image_transport::CameraPublisher image_detected_pub_;
  // For publishing subtracted image
  image_transport::CameraPublisher image_background_subtracted_pub_;

  bool first_image_received_;
  cv::Mat accumulated_image_;

  cv::Ptr<cv::BackgroundSubtractorMOG2> bg_subtractor_; // With regard to shadows
  double moving_average_weight_ = 1.0;
  int activation_threshold_ = 170;
  bool automatic_learning_rate_ = false; // Set learning rate automatically
  double learning_rate_ = 0.7;           // Learning rate of bg subtractor
  int detectionLimit_ = 4; // The maximal number of detections to make/objects to track
  int min_area_ = 60;      // To filter smaller areas
  int max_area_ = 5000;    // To filter bigger areas
  int erosion_iterations_ = 2;
  int dilation_iterations_ = 10; // For controlling the iterations of erosion/dilation
  bool shadows_ = false;         // Control if shadows should be tracked
  bool debug_contours_ = false;
};

} // namespace hector_motion_detection

#endif
