//
// Created by finn on 3/13/25.
//

#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <image_transport/image_transport.hpp>
#include <image_transport/camera_publisher.hpp>

class CameraDummyNode
{
private:
  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<rclcpp::ParameterEventHandler> parameter_event_handler_;
  image_transport::ImageTransport image_transport_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  image_transport::CameraPublisher camera_publisher_;

  std::string image_path_;
  cv::Mat camera_image_;

public:
  explicit CameraDummyNode(const rclcpp::Node::SharedPtr& node) : node_(node), image_transport_(node)
  {
    RCLCPP_INFO(node_->get_logger(), "Starting Node");

    parameter_event_handler_ = std::make_shared<rclcpp::ParameterEventHandler>(node_);

    node_->declare_parameter("image_path", "/home/finn/Documents/S11/hector/hector_ws/src/hector_vision/hector_tag_detection/test/Images/image1.jpg");
    image_path_ = node_->get_parameter("image_path").as_string();
    camera_image_ = cv::imread(image_path_);

    camera_publisher_ = image_transport_.advertiseCamera("image", 10);
    publish_timer_ = node_->create_wall_timer(std::chrono::seconds(5), std::bind(&CameraDummyNode::PublishCameraImage , this));

    RCLCPP_INFO(node_->get_logger(), "Node Started");
  }

private:
  void PublishCameraImage() const
  {
    std_msgs::msg::Header header;
    header.frame_id = "dummy_camera";
    header.stamp = node_->now();

    // The info doesn't matter for now
    sensor_msgs::msg::CameraInfo camera_info;
    camera_info.header = header;

    sensor_msgs::msg::Image image;
    cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, camera_image_).toImageMsg(image);
    image.header = header;

    camera_publisher_.publish(image, camera_info);
  }

  void ImagePathParameterCallback(const rclcpp::Parameter& param)
  {
    const auto& new_image_path = param.as_string();
    RCLCPP_DEBUG_STREAM(node_->get_logger(), "Updating image path to: " << new_image_path);
    image_path_ = new_image_path;

    camera_image_ = cv::imread(image_path_);
  }
};

int main( int argc, char **argv )
{
  rclcpp::init(argc, argv);

  const auto node = std::make_shared<rclcpp::Node>("camera_dummy_node");

  auto camera_dummy_node = CameraDummyNode(node);

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}

