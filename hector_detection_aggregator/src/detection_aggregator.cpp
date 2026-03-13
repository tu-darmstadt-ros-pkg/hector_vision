#include <cv_bridge/cv_bridge.hpp>
#include <hector_detection_aggregator/detection_aggregator.hpp>
#include <image_transport/image_transport.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <hector_detection_aggregator/complete_data_detection_aggregator.hpp>
#include <hector_detection_aggregator/detection_aggregator_base.hpp>
#include <hector_detection_aggregator/newest_data_detection_aggregator.hpp>

rclcpp::Duration durationFromDouble( const double &duration )
{
  const int duration_seconds = static_cast<int>( duration );
  const uint32_t duration_nanoseconds = static_cast<int>( ( duration - duration_seconds ) * 1e9 );

  return { duration_seconds, duration_nanoseconds };
}

namespace hector_detection_aggregator
{
DetectionAggregator::DetectionAggregator( const rclcpp::NodeOptions &options )
    : Node( "detection_aggregator", options ), has_subscribers_( false )
{
  declare_parameter<int>( "buffer_size", 16 );
  declare_parameter<double>( "storage_duration", 1.5 );
  declare_parameter<std::string>( "detection_topic", "detection/visual_detection" );
  declare_parameter<std::string>( "aggregation_topic", "detection/aggregated_detections_image" );
  declare_parameter<std::string>( "aggregation_mode", "NEWEST" );
  // Parameter for image transport
  declare_parameter<std::string>( "image_transport", "raw" );

  detection_topic_ = "detection/visual_detection";
  real_detection_topic_ = detection_topic_;
  robot_namespace_ = get_parameter_or<std::string>( "robot_namespace", "" );

  // Color mappings A color for "unknown" is required to exist
  color_map_["motion"] = cv::Scalar( 0, 0, 255 );     // Red
  color_map_["qr"] = cv::Scalar( 255, 0, 0 );         // Blue
  color_map_["apriltag"] = cv::Scalar( 216, 0, 134 ); // Purple
  color_map_["heat"] = cv::Scalar( 0, 255, 0 );       // Green
  color_map_["hazmat"] = cv::Scalar( 255, 255, 0 );   // Turquoise
  color_map_["unknown"] = cv::Scalar( 50, 50, 50 );   // Grey

  RCLCPP_INFO( get_logger(), "Node started" );

  start_image_transfer_timer_ = create_wall_timer( std::chrono::milliseconds( 500 ),
                                                   [this] { this->startImageTransferCallback(); } );
}

void DetectionAggregator::startImageTransferCallback()
{
  start_image_transfer_timer_.reset();

  const std::string aggregation_topic = get_parameter( "aggregation_topic" ).as_string();

  image_transport_ = std::make_shared<image_transport::ImageTransport>( shared_from_this() );
  image_detected_pub_ = image_transport_->advertise( "detection/aggregated_detections_image", 10 );

  image_transport_ = std::make_shared<image_transport::ImageTransport>( shared_from_this() );

  image_detected_pub_ = image_transport_->advertise( "detection/aggregated_detections_image", 10 );

  // Create detection aggregator
  std::string aggregation_mode = get_parameter( "aggregation_mode" ).as_string();
  std::transform( aggregation_mode.begin(), aggregation_mode.end(), aggregation_mode.begin(),
                  ::toupper );
  if ( aggregation_mode == "COMPLETE" ) {
    std::vector<rclcpp::TopicEndpointInfo> publisher_info =
        get_publishers_info_by_topic( real_detection_topic_ );
    size_t buffer_size = get_parameter( "buffer_size" ).as_int();
    detection_aggregator_ = std::make_shared<CompleteDataDetectionAggregator>(
        shared_from_this(), buffer_size, publisher_info );
  } else {
    rclcpp::Duration storage_duration =
        durationFromDouble( get_parameter( "storage_duration" ).as_double() );
    detection_aggregator_ =
        std::make_shared<NewestDataDetectionAggregator>( shared_from_this(), storage_duration );
  }

  check_environment_timer_ =
      create_wall_timer( std::chrono::seconds( 1 ), [this] { this->checkEnvironmentCallback(); } );
}

void DetectionAggregator::createImage()
{
  auto [image, detections] = detection_aggregator_->GetAggregatedData();

  RCLCPP_DEBUG( get_logger(), "Creating detection Image for %lu detections", detections.size() );

  cv::Mat img_detected;
  image->image.copyTo( img_detected );

  for ( const auto &[type, id, detection] : detections ) {
    const cv::Point detection_top_left_point(
        static_cast<int>( detection.bbox.center.position.x - detection.bbox.size_x / 2 ),
        static_cast<int>( detection.bbox.center.position.y - detection.bbox.size_y / 2 ) );
    const cv::Size detection_size( static_cast<int>( detection.bbox.size_x ),
                                   static_cast<int>( detection.bbox.size_y ) );

    const cv::Rect rect( detection_top_left_point, detection_size );

    // Detections of unknown type are colored grey
    cv::Scalar detection_color;
    const auto color_find = color_map_.find( type );
    if ( color_find == color_map_.end() )
      detection_color = color_map_["unknown"];
    else
      detection_color = color_find->second;

    const cv::Point text_point = detection_top_left_point + cv::Point( 0, -12 );
    int text_background_offset;

    const std::string detection_text = type + ": " + id;
    cv::Size text_size = cv::getTextSize( detection_text, cv::FONT_HERSHEY_SIMPLEX, 0.75, 2,
                                          &text_background_offset );
    text_size.height *= 2;

    // Detection marker
    cv::rectangle( img_detected, rect, detection_color, 2, cv::LINE_AA );
    // Text background
    cv::rectangle( img_detected,
                   cv::Rect( text_point + cv::Point( 0, -3 * text_background_offset ), text_size ),
                   cv::Scalar( 255, 255, 255 ), cv::FILLED );
    // Detection id text
    cv::putText( img_detected, detection_text, text_point, cv::FONT_HERSHEY_SIMPLEX, 0.75,
                 detection_color, 2 );
  }

  cv_bridge::CvImage cvImg;
  img_detected.copyTo( cvImg.image );
  // cvImg.header = img->header;
  cvImg.encoding = sensor_msgs::image_encodings::BGR8;
  const auto info = std::make_shared<sensor_msgs::msg::CameraInfo>();
  info->header = image->header;

  RCLCPP_DEBUG( get_logger(), "Publishing detection image" );

  image_detected_pub_.publish( cvImg.toImageMsg() );
}

void DetectionAggregator::imageDetectionCallback( const Detection2DArray::ConstSharedPtr &percept,
                                                  const rclcpp::MessageInfo &info )
{
  {
    const size_t perceptions = percept->detections.size();
    RCLCPP_DEBUG_STREAM( get_logger(), "Aggregating "
                                           << perceptions << " Perception"
                                           << ( perceptions != 1 ? "s" : "" ) << " from "
                                           << info.get_rmw_message_info().publisher_gid.data );
  }

  if ( detection_aggregator_->AddDetection( percept, info ) )
    createImage();
}

void DetectionAggregator::imageCallback( const sensor_msgs::msg::Image::ConstSharedPtr &image )
{
  if ( detection_aggregator_->AddImage( image ) )
    createImage();
}

void DetectionAggregator::startSubscribers()
{
  std::vector<rclcpp::TopicEndpointInfo> publishers = get_publishers_info_by_topic( "/image" );
  RCLCPP_INFO( get_logger(), "Starting subscribers" );
  image_subscriber_ =
      image_transport_->subscribe( "/image", 1, &DetectionAggregator::imageCallback, this );
  image_percept_sub_ = create_subscription<Detection2DArray>(
      robot_namespace_ + "/" + detection_topic_, 1,
      std::bind( &DetectionAggregator::imageDetectionCallback, this, std::placeholders::_1,
                 std::placeholders::_2 ) );

  real_detection_topic_ = image_percept_sub_->get_topic_name();
}

void DetectionAggregator::stopSubscribers()
{
  RCLCPP_INFO( get_logger(), "Stopping subscribers" );
  image_subscriber_.shutdown();
  image_percept_sub_.reset();
}

void DetectionAggregator::checkPublisherSubscriptions()
{
  const size_t subscribers = image_detected_pub_.getNumSubscribers();

  RCLCPP_DEBUG_STREAM( get_logger(), "Subscribers: " << subscribers
                                                     << " has_subscribers_: " << has_subscribers_ );

  // Changed to no subscribers
  if ( subscribers == 0 && has_subscribers_ ) {
    has_subscribers_ = false;
    stopSubscribers();
  }
  // Changed from no subscribers
  if ( subscribers > 0 && !has_subscribers_ ) {
    has_subscribers_ = true;
    startSubscribers();
  }
}

void DetectionAggregator::checkEnvironmentCallback()
{
  checkPublisherSubscriptions();
  const std::vector<rclcpp::TopicEndpointInfo> detector_info =
      get_publishers_info_by_topic( real_detection_topic_ );
  detection_aggregator_->UpdatePublishers( detector_info );
}

} // namespace hector_detection_aggregator

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE( hector_detection_aggregator::DetectionAggregator );
