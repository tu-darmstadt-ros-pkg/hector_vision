#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <hector_detection_aggregator/detection_aggregator.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

namespace hector_detection_aggregator
{
DetectionAggregator::DetectionAggregator(const rclcpp::Node::SharedPtr& node)
  : storage_duration_(0, 1e8), has_subscribers_(false)
{
  node_ = node;
  param_subscriber_ = std::make_shared<rclcpp::ParameterEventHandler>(node_);

  node_->declare_parameter<double>("storage_duration", 10.0);
  auto storage_duration_parameter_callback = [this](const  rclcpp::Parameter& p)
  {
    const double storage_duration_param = node_->get_parameter("storage_duration").as_double();
    const int storage_duration_seconds = static_cast<int>(storage_duration_param);
    const int storage_duration_nanoseconds = static_cast<int>((storage_duration_param - storage_duration_seconds) * 1e9);
    this->storage_duration_ = rclcpp::Duration(storage_duration_seconds, storage_duration_nanoseconds);
  };
  storage_duration_callback_handle_ = param_subscriber_->add_parameter_callback("storage_duration", storage_duration_parameter_callback);

  // Color mappings A color for "unknown" is required to exist
  color_map_["motion"] = cv::Scalar(0,0,255);       // Red
  color_map_["qr"] = cv::Scalar(255,0,0);           // Blue
  color_map_["apriltag"] = cv::Scalar(216, 0, 134); // Purple
  color_map_["heat"] = cv::Scalar(0,255,0);         // Green
  color_map_["hazmat"] = cv::Scalar(255,255,0);     // Turquoise
  color_map_["unknown"] = cv::Scalar(50, 50, 50);   // Grey

  image_transport_ = std::make_shared<image_transport::ImageTransport>(node_);
  image_transport::SubscriberStatusCallback connect_cb = std::bind(
    &DetectionAggregator::connectCb, this);

  image_detected_pub_ = image_transport_->advertiseCamera(
    "/detection/aggregated_detections_image", 10);

  check_subscribers_timer_ = node_->create_wall_timer(std::chrono::seconds(1), std::bind(&DetectionAggregator::publisherSubscriptionCallback, this));

  current_color_image_.reset();
  image_percept_sub_.reset();

  RCLCPP_INFO(node_->get_logger(), "Node started");
}

DetectionAggregator::~DetectionAggregator() = default;

void DetectionAggregator::updateDetections()
{
  rclcpp::Time storage_threshold;
  if ((node_->now().seconds() > storage_duration_.seconds())){
    storage_threshold = node_->now() - storage_duration_;
  }

  for (auto it = detection_map_.begin(); it != detection_map_.end(); )
  {
    const auto current_it = it;
    ++it;
    if (current_it->second.header.stamp.sec < storage_threshold.seconds())
    {
      detection_map_.erase(current_it);
    }
  }
}

void DetectionAggregator::createImage()
{
  updateDetections();
  RCLCPP_DEBUG(node_->get_logger(), "Creating detection Image for %lu detections", detection_map_.size());

  if (!current_color_image_)
    return;

  cv::Mat img_detected;
  current_color_image_->image.copyTo(img_detected);

  for (const auto& [id, detection] : detection_map_)
  {
    const cv::Point detection_center_point(static_cast<int>(detection.bbox.center.position.x - detection.bbox.size_x / 2),
                                           static_cast<int>(detection.bbox.center.position.y - detection.bbox.size_y / 2));
    const cv::Size detection_size(static_cast<int>(detection.bbox.size_x),
                                  static_cast<int>(detection.bbox.size_y));

    const cv::Rect rect(detection_center_point, detection_size);

    // Detections of unknown type are colored grey
    cv::Scalar detection_color;
    const auto color_find = color_map_.find(id.first);
    if (color_find == color_map_.end())
      detection_color = color_map_["unknown"];
    else
      detection_color = color_find->second;

    const std::string detection_text = id.first + ": " + id.second;

    //ROS_INFO("Type: %s color %f %f %f ",percept_pair.first.c_str(),color_map_[percept_pair.first][0],color_map_[percept_pair.first][1],color_map_[percept_pair.first][2]);
    cv::rectangle(img_detected, rect, detection_color, 2, cv::LINE_AA);
    cv::putText(img_detected, detection_text, detection_center_point + cv::Point(0, -10), cv::FONT_HERSHEY_SIMPLEX, 0.75, detection_color, 2);
  }

  cv_bridge::CvImage cvImg;
  img_detected.copyTo(cvImg.image);
  //cvImg.header = img->header;
  cvImg.encoding = sensor_msgs::image_encodings::BGR8;
  const auto info = std::make_shared<sensor_msgs::msg::CameraInfo>();
  info->header = current_color_image_->header;

  RCLCPP_DEBUG(node_->get_logger(), "Publishing detection image");

  image_detected_pub_.publish(cvImg.toImageMsg(), current_camera_info_);
}

void DetectionAggregator::imageDetectionCallback(const Detection2DArray::ConstSharedPtr& percept)
{
  RCLCPP_DEBUG(node_->get_logger(), "Aggregating Perceptions");

  // Write all detections with identifications into the map
  for (const auto& detection : percept->detections)
  {
    vision_msgs::msg::ObjectHypothesis likeliest_type;
    likeliest_type.class_id = "nothing";
    likeliest_type.score = 0.0;

    for (const auto& hypothesis : detection.results)
    {
      const auto& type_hypothesis = hypothesis.hypothesis;
      if (type_hypothesis.score > likeliest_type.score)
        likeliest_type = type_hypothesis;
    }

    if (likeliest_type.class_id == "nothing")
      continue;

    detection_map_[{likeliest_type.class_id, detection.id}] = detection;
  }
}

void DetectionAggregator::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& image,
                                        const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info)
{
  if(image_detected_pub_.getNumSubscribers() == 0)
    return;

  // current_grey_image_ = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::MONO8);
  current_color_image_ = cv_bridge::toCvShare(image, sensor_msgs::image_encodings::BGR8);
  current_camera_info_ = camera_info;

  createImage();
}

void DetectionAggregator::connectCb()
{
  if (image_detected_pub_.getNumSubscribers() == 0) {
    stopSubscribers();
    RCLCPP_INFO(node_->get_logger(), "Stopping subscribers");
  } else {
    startSubscribers();
    RCLCPP_INFO(node_->get_logger(), "Starting subscribers");
  }
}

void DetectionAggregator::startSubscribers()
{
  RCLCPP_INFO(node_->get_logger(), "Starting subscribers");
  camera_subscriber_ = image_transport_->subscribeCamera("/image", 1, &DetectionAggregator::imageCallback, this);
  image_percept_sub_ = node_->create_subscription<Detection2DArray>(
    "/detection/visual_detection", 1,
    std::bind(&DetectionAggregator::imageDetectionCallback, this, std::placeholders::_1));
}

void DetectionAggregator::stopSubscribers()
{
  RCLCPP_INFO(node_->get_logger(), "Stopping subscribers");
  camera_subscriber_.shutdown();
  image_percept_sub_.reset();
}

void DetectionAggregator::publisherSubscriptionCallback()
{
  const size_t subscribers = image_detected_pub_.getNumSubscribers();

  RCLCPP_DEBUG_STREAM(node_->get_logger(), "Subscribers: " << subscribers << " has_subscribers_: " << has_subscribers_);

  // Changed to no subscribers
  if (subscribers == 0 && has_subscribers_)
  {
    has_subscribers_ = false;
    stopSubscribers();
  }
  // Changed from no subscribers
  if (subscribers > 0 && !has_subscribers_)
  {
    has_subscribers_ = true;
    startSubscribers();
  }
}
}

int main( int argc, char **argv )
{
  rclcpp::init( argc, argv );

  const auto node = std::make_shared<rclcpp::Node>("detection_aggregator");
  auto detection_aggregator = hector_detection_aggregator::DetectionAggregator(node);

  rclcpp::spin( node );

  rclcpp::shutdown();
  return 0;
}
