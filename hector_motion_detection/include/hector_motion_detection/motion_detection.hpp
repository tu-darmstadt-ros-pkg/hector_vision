#ifndef HECTOR_MOTION_DETECTION_HPP
#define HECTOR_MOTION_DETECTION_HPP

#include <hector_ros2_utils/node.hpp>

#include <image_transport/image_transport.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

namespace hector_motion_detection
{

class MotionDetection : public hector::Node
{
public:
  MotionDetection( const rclcpp::NodeOptions &options );
  MotionDetection( MotionDetection &md ) = delete;
  MotionDetection( MotionDetection &&md ) = delete;

private:
  void imageCallback( const sensor_msgs::msg::Image::ConstSharedPtr &img );
  void debugPublisherCallback( const bool &enabled );
  void publisherSubscriptionCallback();

  void startSubscribers();
  void stopSubscribers();

  bool has_subscribers_;
  bool first_image_received_;
  cv::Mat accumulated_image_;

  std::shared_ptr<image_transport::ImageTransport> image_transport_;
  image_transport::Subscriber image_sub_;

  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr image_perception_pub_;

  rclcpp::TimerBase::SharedPtr check_subscriptions_timer_;

  // Debug image publishers
  image_transport::CameraPublisher image_motion_pub_;
  image_transport::CameraPublisher image_detected_pub_;
  image_transport::CameraPublisher image_background_subtracted_pub_;

  // Reconfigurable properties
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
  bool debug_images_ = false;
};

} // namespace hector_motion_detection

#endif
