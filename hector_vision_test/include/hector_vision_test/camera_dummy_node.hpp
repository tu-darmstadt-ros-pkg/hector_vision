//
// Created by finn on 4/24/25.
//

#ifndef CAMERA_DUMMY_NODE_HPP
#define CAMERA_DUMMY_NODE_HPP

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>

class CameraDummyNode
{
private:
  std::string image_dir_;
  std::vector<cv::Mat> images_;
  size_t current_image_;
  int image_frames_;
  int current_image_frame_;

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<rclcpp::ParameterEventHandler> parameter_event_handler_;
  image_transport::ImageTransport image_transport_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  image_transport::CameraPublisher camera_publisher_;

public:
  explicit CameraDummyNode( const rclcpp::Node::SharedPtr &node );

private:
  void publishImage();

  void iterateImage();
};

#endif // CAMERA_DUMMY_NODE_HPP
