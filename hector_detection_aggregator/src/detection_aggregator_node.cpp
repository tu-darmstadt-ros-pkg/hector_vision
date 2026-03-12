#include <cv_bridge/cv_bridge.hpp>
#include <hector_detection_aggregator/detection_aggregator_node.hpp>
#include <image_transport/image_transport.hpp>
#include <memory>
#include <opencv2/highgui/highgui.hpp>
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
DetectionAggregatorNode::DetectionAggregatorNode( const rclcpp::Node::SharedPtr &node )
    : enabled_( true ), has_subscribers_( false )
{
  node_ = node;
  param_subscriber_ = std::make_shared<rclcpp::ParameterEventHandler>( node_ );

  node_->declare_parameter<bool>( "enabled", true );
  node_->declare_parameter<int>( "buffer_size", 16 );
  node_->declare_parameter<double>( "storage_duration", 1.5 );
  node_->declare_parameter<std::string>( "detection_topic", "detection/visual_detection" );
  node_->declare_parameter<std::string>( "aggregation_topic",
                                         "detection/aggregated_detections_image" );
  node_->declare_parameter<std::string>( "aggregation_mode", "NEWEST" );
  // Parameter for image transport
  node_->declare_parameter<std::string>( "image_transport", "raw" );

  enabled_callback_handle_ = param_subscriber_->add_parameter_callback(
      "enabled", [this]( const rclcpp::Parameter &parameter ) -> void {
        RCLCPP_DEBUG( this->node_->get_logger(), "Parameter callback for \"enabled\" called" );
        this->enabledCallback( parameter.as_bool() );
      } );
  detection_topic_ = node_->get_parameter( "detection_topic" ).as_string();
  real_detection_topic_ = detection_topic_;
  const std::string aggregation_topic = node_->get_parameter( "aggregation_topic" ).as_string();
  robot_namespace_ = node_->get_parameter_or<std::string>( "robot_namespace", "" );

  // Color mappings A color for "unknown" is required to exist
  color_map_["motion"] = cv::Scalar( 0, 0, 255 );     // Red
  color_map_["qr"] = cv::Scalar( 255, 0, 0 );         // Blue
  color_map_["apriltag"] = cv::Scalar( 216, 0, 134 ); // Purple
  color_map_["heat"] = cv::Scalar( 0, 255, 0 );       // Green
  color_map_["hazmat"] = cv::Scalar( 255, 255, 0 );   // Turquoise
  color_map_["unknown"] = cv::Scalar( 50, 50, 50 );   // Grey

  image_transport_ = std::make_shared<image_transport::ImageTransport>( node_ );

  image_detected_pub_ = image_transport_->advertise( robot_namespace_ + "/" + aggregation_topic, 10 );

  check_environment_timer_ = node_->create_wall_timer(
      std::chrono::seconds( 1 ),
      std::bind( &DetectionAggregatorNode::checkEnvironmentCallback, this ) );

  image_percept_sub_.reset();

  enabled_sub_ = node_->create_subscription<std_msgs::msg::Bool>(
      "detection_aggregator/enabled", 10,
      std::bind( &DetectionAggregatorNode::msgEnabledCallback, this, std::placeholders::_1 ) );
  enabled_pub_ =
      node_->create_publisher<std_msgs::msg::Bool>( "detection_aggregator/enabled_status", 10 );

  // Create detection aggregator
  std::string aggregation_mode = node_->get_parameter( "aggregation_mode" ).as_string();
  std::transform( aggregation_mode.begin(), aggregation_mode.end(), aggregation_mode.begin(),
                  ::toupper );
  if ( aggregation_mode == "COMPLETE" ) {
    std::vector<rclcpp::TopicEndpointInfo> publisher_info =
        node_->get_publishers_info_by_topic( real_detection_topic_ );
    size_t buffer_size = node_->get_parameter( "buffer_size" ).as_int();
    detection_aggregator_ =
        std::make_shared<CompleteDataDetectionAggregator>( node_, buffer_size, publisher_info );
  } else {
    rclcpp::Duration storage_duration =
        durationFromDouble( node_->get_parameter( "storage_duration" ).as_double() );
    detection_aggregator_ =
        std::make_shared<NewestDataDetectionAggregator>( node_, storage_duration );
  }

  RCLCPP_INFO( node_->get_logger(), "Node started" );
}

DetectionAggregatorNode::~DetectionAggregatorNode() = default;

void DetectionAggregatorNode::createImage()
{
  auto [image, detections] = detection_aggregator_->GetAggregatedData();

  RCLCPP_DEBUG( node_->get_logger(), "Creating detection Image for %lu detections",
                detections.size() );

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

  RCLCPP_DEBUG( node_->get_logger(), "Publishing detection image" );

  image_detected_pub_.publish( cvImg.toImageMsg() );
}

void DetectionAggregatorNode::imageDetectionCallback( const Detection2DArray::ConstSharedPtr &percept,
                                                      const rclcpp::MessageInfo &info )
{
  {
    const size_t perceptions = percept->detections.size();
    RCLCPP_DEBUG_STREAM( node_->get_logger(),
                         "Aggregating " << perceptions << " Perception"
                                        << ( perceptions != 1 ? "s" : "" ) << " from "
                                        << info.get_rmw_message_info().publisher_gid.data );
  }

  if ( detection_aggregator_->AddDetection( percept, info ) )
    createImage();
}

void DetectionAggregatorNode::imageCallback( const sensor_msgs::msg::Image::ConstSharedPtr &image )
{
  if ( detection_aggregator_->AddImage( image ) )
    createImage();
}

void DetectionAggregatorNode::publishEnableStatus() const
{
  std_msgs::msg::Bool bool_msg;
  bool_msg.data = enabled_;
  enabled_pub_->publish( bool_msg );

  RCLCPP_INFO_STREAM( node_->get_logger(), ( enabled_ ? "Enabled" : "Disabled" )
                                               << " tag_detection." );
}

void DetectionAggregatorNode::enabledCallback( const bool &enabled )
{
  // Changed to disabled
  if ( !enabled && enabled_ ) {
    enabled_ = false;
    if ( has_subscribers_ )
      stopSubscribers();
  }
  // Changed to enabled
  if ( enabled && !enabled_ ) {
    enabled_ = true;
    if ( has_subscribers_ )
      startSubscribers();
  }

  publishEnableStatus();
}

void DetectionAggregatorNode::msgEnabledCallback( const std_msgs::msg::Bool::ConstSharedPtr &enabled ) const
{
  const rclcpp::Parameter parameter( "enabled", rclcpp::ParameterValue( enabled->data ) );
  node_->set_parameter( parameter );
}

void DetectionAggregatorNode::startSubscribers()
{
  std::vector<rclcpp::TopicEndpointInfo> publishers =
      node_->get_publishers_info_by_topic( "/image" );
  RCLCPP_INFO( node_->get_logger(), "Starting subscribers" );
  image_subscriber_ =
      image_transport_->subscribe( "/image", 1, &DetectionAggregatorNode::imageCallback, this );
  image_percept_sub_ = node_->create_subscription<Detection2DArray>(
      robot_namespace_ + "/" + detection_topic_, 1,
      std::bind( &DetectionAggregatorNode::imageDetectionCallback, this, std::placeholders::_1,
                 std::placeholders::_2 ) );

  real_detection_topic_ = image_percept_sub_->get_topic_name();
}

void DetectionAggregatorNode::stopSubscribers()
{
  RCLCPP_INFO( node_->get_logger(), "Stopping subscribers" );
  image_subscriber_.shutdown();
  image_percept_sub_.reset();
}

void DetectionAggregatorNode::checkPublisherSubscriptions()
{
  const size_t subscribers = image_detected_pub_.getNumSubscribers();

  RCLCPP_DEBUG_STREAM( node_->get_logger(),
                       "Subscribers: " << subscribers << " has_subscribers_: " << has_subscribers_ );

  // Changed to no subscribers
  if ( subscribers == 0 && has_subscribers_ ) {
    has_subscribers_ = false;
    if ( enabled_ )
      stopSubscribers();
  }
  // Changed from no subscribers
  if ( subscribers > 0 && !has_subscribers_ ) {
    has_subscribers_ = true;
    if ( enabled_ )
      startSubscribers();
  }
}

void DetectionAggregatorNode::checkEnvironmentCallback()
{
  checkPublisherSubscriptions();
  const std::vector<rclcpp::TopicEndpointInfo> detector_info =
      node_->get_publishers_info_by_topic( real_detection_topic_ );
  detection_aggregator_->UpdatePublishers( detector_info );
}

} // namespace hector_detection_aggregator

int main( int argc, char **argv )
{
  rclcpp::init( argc, argv );

  const auto node = std::make_shared<rclcpp::Node>( "detection_aggregator" );
  auto detection_aggregator = hector_detection_aggregator::DetectionAggregatorNode( node );

  rclcpp::spin( node );

  rclcpp::shutdown();
  return 0;
}
