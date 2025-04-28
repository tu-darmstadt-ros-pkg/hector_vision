//
// Created by finn on 3/13/25.
//

#include <string>
#include <filesystem>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <image_transport/camera_publisher.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <hector_vision_test/camera_dummy_node.hpp>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

CameraDummyNode::CameraDummyNode(const rclcpp::Node::SharedPtr& node)
: current_image_(0), node_(node), image_transport_(node), current_image_frame_(0)
{
  RCLCPP_INFO(node_->get_logger(), "Starting Node");

  parameter_event_handler_ = std::make_shared<rclcpp::ParameterEventHandler>(node_);

  node_->declare_parameter("image_dir",
    ament_index_cpp::get_package_share_directory( "hector_vision_test" ) + "/images");
  node_->declare_parameter("image_frequency", 0.2);
  node_->declare_parameter("image_frames", 25);

  image_dir_ = node_->get_parameter("image_dir").as_string();
  const double image_frequency = node_->get_parameter("image_frequency").as_double();
  image_frames_ = static_cast<int>(node_->get_parameter("image_frames").as_int());

  uint file_count = 0;
  for (const auto& file : fs::directory_iterator(image_dir_))
    ++file_count;

  try
  {
    images_.reserve(file_count);
  }
  catch (std::bad_alloc& e)
  {
    RCLCPP_ERROR_STREAM(node_->get_logger(), "Unable to allocate memory for images: " << e.what());
    throw;
  }
  for (const auto& file : fs::directory_iterator(image_dir_))
  {
    try {
      images_.push_back(cv::imread(file.path()));
    }
    catch(std::exception& e)
    {
      RCLCPP_ERROR_STREAM(node_->get_logger(), "Could not load image file " << file.path() << ": " << e.what());
      throw;
    }
  }

  camera_publisher_ = image_transport_.advertiseCamera("image", 10);

  publish_timer_ = node_->create_wall_timer(
    std::chrono::milliseconds(static_cast<int>(1000 / (image_frequency * static_cast<double>(image_frames_)))),
    std::bind(&CameraDummyNode::publishImage, this));

  RCLCPP_INFO(node_->get_logger(), "Node Started");
}

void CameraDummyNode::iterateImage()
{
  ++current_image_;
  if (current_image_ >= images_.size())
    current_image_ = 0;

  RCLCPP_INFO_STREAM(node_->get_logger(), "Iterating image to " << current_image_);
}


void CameraDummyNode::publishImage()
{
  if (current_image_frame_ == image_frames_) {
    iterateImage();
    current_image_frame_ = 0;
  }

  std_msgs::msg::Header header;
  header.frame_id = "dummy_camera";
  header.stamp = node_->now();

  // The info doesn't matter for now
  sensor_msgs::msg::CameraInfo camera_info;
  camera_info.header = header;

  sensor_msgs::msg::Image image;
  cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, images_[current_image_]).toImageMsg(image);
  image.header = header;

  camera_publisher_.publish(image, camera_info);

  ++current_image_frame_;
}

int main( int argc, char **argv )
{
  rclcpp::init(argc, argv);

  const auto node = std::make_shared<rclcpp::Node>("camera_dummy_node");

  auto camera_dummy_node = CameraDummyNode(node);

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}

