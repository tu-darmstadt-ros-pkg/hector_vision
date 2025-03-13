//=================================================================================================
// Copyright (c) 2012, Johannes Meyer, TU Darmstadt
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the Flight Systems and Automatic Control group,
//       TU Darmstadt, nor the names of its contributors may be used to
//       endorse or promote products derived from this software without
//       specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//=================================================================================================

#include <limits>
#include <rclcpp/rclcpp.hpp>
#include <hector_qrcode_detection/qrcode_detection.hpp>
#include <tf2/LinearMath/Scalar.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <zbar.h>

using namespace zbar;

namespace hector_qrcode_detection {

QrcodeDetectionImpl::QrcodeDetectionImpl(const rclcpp::Node::SharedPtr& node)
  : node_(node), image_transport_(node), has_subscribers_(false)
{
  RCLCPP_INFO(node_->get_logger(), "qrcode init");

  scanner_ = new zbar::ImageScanner;
  scanner_->set_config(ZBAR_QRCODE, ZBAR_CFG_ENABLE, 1);

  node_->declare_parameter("enabled", true);

  enabled_ = node->get_parameter("enabled").as_bool();

  // worldmodel_percept_publisher_ = nh_.advertise<hector_worldmodel_msgs::ImagePercept>("image_percept", 10);
  rclcpp::PublisherOptions qrcode_pub_options;
  qrcode_image_publisher_ = image_transport_.advertiseCamera(
    "image/qrcode", 10);
  rclcpp::PublisherOptions aggregator_percept_pub_options;
  aggregator_percept_publisher_ = node_->create_publisher<Detection2DArray>(
    "perception/image_percept", 10);

  check_subscribers_timer_ = node_->create_wall_timer(std::chrono::seconds(1), std::bind(&QrcodeDetectionImpl::publisherSubscriptionCallback, this));

  enabled_sub_ = node_->create_subscription<std_msgs::msg::Bool>(
    "enabled", 10, std::bind(&QrcodeDetectionImpl::enabledCallback, this, std::placeholders::_1));
  enabled_pub_ = node_->create_publisher<std_msgs::msg::Bool>("enabled_status", 10);
  
  publishEnableStatus();

  RCLCPP_INFO(node_->get_logger(), "Successfully initialized the zbar qrcode detector for image %s", camera_subscriber_.getTopic().c_str());
}

void QrcodeDetectionImpl::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& image,
                                        const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info)
{
  cv_bridge::CvImageConstPtr cv_image;
  cv_image = cv_bridge::toCvShare(image, "mono8");
  cv::Mat rotation_matrix = cv::Mat::eye(2,3,CV_32FC1);

  RCLCPP_DEBUG(node_->get_logger(), "Received new image with %u x %u pixels.", image->width, image->height);

  // wrap image data
  Image zbar(cv_image->image.cols, cv_image->image.rows, "Y800", cv_image->image.data, cv_image->image.cols * cv_image->image.rows);

  // scan the image for barcodes
  scanner_->scan(zbar);

  // extract results
  // hector_worldmodel_msgs::ImagePercept worldmodel_percept;
  // worldmodel_percept.header = image->header;
  // worldmodel_percept.camera_info = *camera_info;
  // worldmodel_percept.info.class_id = "qrcode";
  // worldmodel_percept.info.class_support = 1.0;

  Detection2DArray perception_array;
  perception_array.header = image->header;

  for(Image::SymbolIterator symbol = zbar.symbol_begin(); symbol != zbar.symbol_end(); ++symbol)
  {
    RCLCPP_DEBUG_STREAM(node_->get_logger(), "Decoded " << symbol->get_type_name() << " symbol \"" << symbol->get_data() << '"');

    // percept.info.object_id = ros::this_node::getName() + "/" + symbol->get_data();
    // percept.info.object_id = symbol->get_data();
    // worldmodel_percept.info.object_support = 1.0;
    // worldmodel_percept.info.name = symbol->get_data();

    if (symbol->get_location_size() != 4)
    {
      RCLCPP_WARN(node_->get_logger(), "Could not get symbol locations(location_size != 4)");
      continue;
    }

    // point order is left/top, left/bottom, right/bottom, right/top
    int min_x = std::numeric_limits<int>::max(), min_y = std::numeric_limits<int>::max(), max_x = 0, max_y = 0;
    for(int i = 0; i < 4; ++i)
    {
      if (symbol->get_location_x(i) > max_x) max_x = symbol->get_location_x(i);
      if (symbol->get_location_x(i) < min_x) min_x = symbol->get_location_x(i);
      if (symbol->get_location_y(i) > max_y) max_y = symbol->get_location_y(i);
      if (symbol->get_location_y(i) < min_y) min_y = symbol->get_location_y(i);
    }

    vision_msgs::msg::ObjectHypothesis hypothesis;
    hypothesis.class_id = "qr";
    hypothesis.score = 1.0;

    // The Pose estimate is currently not used
    vision_msgs::msg::ObjectHypothesisWithPose identification_pose;
    identification_pose.hypothesis = hypothesis;

    vision_msgs::msg::Point2D bounding_box_center_point;
    bounding_box_center_point.x = (min_x + max_x) / 2.0;
    bounding_box_center_point.y = (min_y + max_y) / 2.0;

    vision_msgs::msg::Pose2D bounding_box_center_pose;
    bounding_box_center_pose.theta = 0.0;
    bounding_box_center_pose.position = bounding_box_center_point;

    vision_msgs::msg::BoundingBox2D bounding_box;
    bounding_box.center = bounding_box_center_pose;
    bounding_box.size_x = max_x - min_x;
    bounding_box.size_y = max_y - min_y;

    vision_msgs::msg::Detection2D perception_data;
    perception_data.header = perception_array.header;
    perception_data.id = symbol->get_data();
    perception_data.results.push_back(identification_pose);
    perception_data.bbox = bounding_box;

    perception_array.detections.push_back(perception_data);

    if (qrcode_image_publisher_.getNumSubscribers() > 0)
    {
      try
      {
        cv::Rect rect(
          cv::Point2i(std::max(min_x, 0), std::max(min_y, 0)),
          cv::Point2i(std::min(max_x, cv_image->image.cols), std::min(max_y, cv_image->image.rows)));

        cv_bridge::CvImagePtr qrcode_cv(new cv_bridge::CvImage(*cv_image));
        qrcode_cv->image = cv_image->image(rect);

        sensor_msgs::msg::Image qrcode_image;
        qrcode_cv->toImageMsg(qrcode_image);
        qrcode_image_publisher_.publish(qrcode_image, *camera_info);
      }
      catch(cv::Exception& e)
      {
        RCLCPP_ERROR(node_->get_logger(), "cv::Exception: %s", e.what());
      }
    }
  }

  if (aggregator_percept_publisher_->get_subscription_count() > 0)
    aggregator_percept_publisher_->publish(perception_array);

  // clean up
  zbar.set_data(nullptr, 0);
}

void QrcodeDetectionImpl::enabledCallback(const std_msgs::msg::Bool::ConstSharedPtr& enabled)
{
  // Changed to disabled
  if (!enabled->data && enabled_)
  {
    enabled_ = false;
    if (has_subscribers_)
      stopSubscribers();
  }
  // Changed to enabled
  if (enabled->data && !enabled_)
  {
    enabled_ = true;
    if (has_subscribers_)
      startSubscribers();
  }
}

void QrcodeDetectionImpl::publisherSubscriptionCallback()
{
  const size_t subscribers = qrcode_image_publisher_.getNumSubscribers()
                           + aggregator_percept_publisher_->get_subscription_count();

  // Changed to no subscribers
  if (subscribers == 0 && has_subscribers_)
  {
    has_subscribers_ = false;
    if (enabled_)
      stopSubscribers();
  }
  // Changed from no subscribers
  if (subscribers > 1 && !has_subscribers_)
  {
    has_subscribers_ = true;
    if (enabled_)
      startSubscribers();
  }
}

void QrcodeDetectionImpl::publishEnableStatus() const
{
  std_msgs::msg::Bool bool_msg;
  bool_msg.data = enabled_;
  enabled_pub_->publish(bool_msg);

  std::string enabled_string;
  
  RCLCPP_INFO_STREAM(node_->get_logger(), (enabled_ ? "Enabled" : "Disabled") << " qrcode_detection.");
}

void QrcodeDetectionImpl::startSubscribers()
{
  RCLCPP_INFO(node_->get_logger(), "Starting subscribers");
  camera_subscriber_ = image_transport_.subscribeCamera("image", 10, &QrcodeDetectionImpl::imageCallback, this);
}

void QrcodeDetectionImpl::stopSubscribers()
{
  RCLCPP_INFO(node_->get_logger(), "Stopping subscribers");
  camera_subscriber_.shutdown();
}
} // namespace hector_qrcode_detection

int main( int argc, char **argv )
{
  rclcpp::init(argc, argv);

  const auto node = std::make_shared<rclcpp::Node>("qrcode_detection");
  auto detection_aggregator = hector_qrcode_detection::QrcodeDetectionImpl(node);

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
