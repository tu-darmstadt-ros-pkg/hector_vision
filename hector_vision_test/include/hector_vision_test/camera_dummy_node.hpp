//
// Created by finn on 4/24/25.
//

#ifndef CAMERA_DUMMY_NODE_HPP
#define CAMERA_DUMMY_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>

class CameraDummyNode
{
private:
  std::string image_dir_;
  std::vector<cv::Mat> images_;
  size_t current_image_;

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<rclcpp::ParameterEventHandler> parameter_event_handler_;
  image_transport::ImageTransport image_transport_;
  rclcpp::TimerBase::SharedPtr image_timer_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  image_transport::CameraPublisher camera_publisher_;

public:
  explicit CameraDummyNode(const rclcpp::Node::SharedPtr& node);

private:
  void publishImage() const;

  void iterateImage();
};

#endif //CAMERA_DUMMY_NODE_HPP
