#include <hector_detection_aggregator/detection_aggregator.hpp>

namespace hector_detection_aggregator
{

DetectionAggregator::DetectionAggregator() : Node( "detection_aggregator" )
{
  color_map_["motion"] = cv::Scalar( 0, 0, 255 );   // Red
  color_map_["QR"] = cv::Scalar( 255, 0, 0 );       // Blue
  color_map_["april"] = cv::Scalar( 216, 0, 134 );  // Purple
  color_map_["thermal"] = cv::Scalar( 0, 255, 0 );  // Green
  color_map_["hazmat"] = cv::Scalar( 255, 255, 0 ); // Turquoise
}

bool DetectionAggregator::readParameters()
{
  param_listener_ = std::make_shared<detection_aggregator::ParamListener>( this );
  if ( !param_listener_ ) {
    RCLCPP_ERROR( get_logger(), "Parameters could be loaded" );
    return false;
  }
  params_ = std::make_shared<detection_aggregator::Params>( param_listener_->get_params() );

  if ( params_->visualization_topic.empty() ) {
    RCLCPP_ERROR( get_logger(), "Visualization topic needs to be specified" );
    return false;
  }

  if ( params_->cam_topic.empty() ) {
    RCLCPP_ERROR( get_logger(), "Cam topic needs to be specified" );
    return false;
  }

  if ( params_->use_thermal && params_->thermal_topic.empty() ) {
    RCLCPP_ERROR( get_logger(), "If using thermal, thermal topic can't be empty" );
    return false;
  }

  if ( params_->thermal_eps < 0 ) {
    RCLCPP_ERROR( get_logger(), "Thermal eps can't be negative" );
    return false;
  }

  if ( params_->use_motion && params_->motion_topic.empty() ) {
    RCLCPP_ERROR( get_logger(), "If using motion, motion topic can't be empty" );
    return false;
  }

  bool use_obj_detection = params_->use_hazmat || params_->use_qr || params_->use_april;
  if ( use_obj_detection && params_->object_detection_topic.empty() ) {
    RCLCPP_ERROR( get_logger(),
                  "If using QR april tags or hazmat, object detection topic can't be empty" );
    return false;
  }

  return true;
}

bool DetectionAggregator::setup()
{
  bool success = readParameters();
  if ( !success ) {
    RCLCPP_ERROR( get_logger(), "Parameter reading failed. Aborting." );
    return false;
  }

  image_transport_ = std::make_shared<image_transport::ImageTransport>( shared_from_this() );
  time_sync_filter_.init( shared_from_this(),
                          std::bind( &DetectionAggregator::processDetectionSet, this,
                                     std::placeholders::_1, std::placeholders::_2,
                                     std::placeholders::_3, std::placeholders::_4 ),
                          params_, image_transport_ );

  vis_pub_ = image_transport_->advertise( params_->visualization_topic, 1 );

  return true;
}

void DetectionAggregator::processDetectionSet(
    std::shared_ptr<const sensor_msgs::msg::Image> cam_img,
    std::shared_ptr<const sensor_msgs::msg::Image> thermal_img,
    std::shared_ptr<const Detection2DArray> motion_detections,
    std::shared_ptr<const Detection2DArray> obj_detections )
{

  if ( vis_pub_.getNumSubscribers() == 0 )
    return;

  RCLCPP_DEBUG_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "Received full detection set for image timestamp: %d.%09u. Creating visualization.",
      cam_img->header.stamp.sec, cam_img->header.stamp.nanosec );

  cv::Mat vis_img;
  try {
    vis_img = cv_bridge::toCvShare( cam_img, sensor_msgs::image_encodings::BGR8 )->image.clone();
  } catch ( const cv_bridge::Exception &e ) {
    RCLCPP_ERROR( get_logger(), "cv_bridge exception converting camera image: %s", e.what() );
    return;
  }

  // Combine obj and motions detections in a single list. Dummy results are just an empty list.
  std::vector<Detection2D> combined_detections = std::vector<Detection2D>();
  combined_detections.insert( combined_detections.begin(), obj_detections->detections.begin(),
                              obj_detections->detections.end() );
  combined_detections.insert( combined_detections.end(), motion_detections->detections.begin(),
                              motion_detections->detections.end() );

  for ( const auto &[header, id, type, score, bbox, _1, _2] : combined_detections ) {

    // We need to check specifcally for usage of hazmat / april / QR since they will be published to
    // the same topic by the semantic detection pipeline

    if ( type == "QR" && !params_->use_qr )
      continue;
    if ( type == "april" && !params_->use_april )
      continue;
    if ( type == "hazmat" && !params_->use_hazmat )
      continue;

    const cv::Point detection_top_left_point( static_cast<int>( bbox.left ),
                                              static_cast<int>( bbox.top ) );
    const cv::Size detection_size( static_cast<int>( bbox.right - bbox.left ),
                                   static_cast<int>( bbox.bottom - bbox.top ) );

    const cv::Rect rect( detection_top_left_point, detection_size );

    // Detections of unknown type are colored grey
    cv::Scalar detection_color;
    const auto color_find = color_map_.find( type );
    if ( color_find == color_map_.end() )
      // Unknown detection type
      continue;
    else
      detection_color = color_find->second;

    const cv::Point text_point = detection_top_left_point + cv::Point( 0, -12 );
    int text_background_offset;

    const std::string detection_text = type + ": " + std::to_string( id );
    cv::Size text_size = cv::getTextSize( detection_text, cv::FONT_HERSHEY_SIMPLEX, 0.75, 2,
                                          &text_background_offset );
    text_size.height *= 2;

    // Detection marker
    cv::rectangle( vis_img, rect, detection_color, 2, cv::LINE_AA );
    // Text background
    cv::rectangle( vis_img,
                   cv::Rect( text_point + cv::Point( 0, -3 * text_background_offset ), text_size ),
                   cv::Scalar( 255, 255, 255 ), cv::FILLED );
    // Detection id text
    cv::putText( vis_img, detection_text, text_point, cv::FONT_HERSHEY_SIMPLEX, 0.75,
                 detection_color, 2 );
  }

  if ( params_->use_thermal ) {

    cv::Mat cv_thermal_img;
    try {
      cv_thermal_img =
          cv_bridge::toCvShare( thermal_img, sensor_msgs::image_encodings::BGR8 )->image.clone();
    } catch ( const cv_bridge::Exception &e ) {
      RCLCPP_ERROR( get_logger(), "cv_bridge exception converting camera image: %s", e.what() );
      return;
    }

    // Canvas grows to fit both images at their native resolution — neither is resized.
    const int canvas_width = vis_img.cols + cv_thermal_img.cols;
    const int canvas_height = std::max( vis_img.rows, cv_thermal_img.rows );

    cv::Mat split_screen( canvas_height, canvas_width, vis_img.type(), cv::Scalar( 0, 0, 0 ) );

    // Camera image: untouched, top-left.
    vis_img.copyTo( split_screen( cv::Rect( 0, 0, vis_img.cols, vis_img.rows ) ) );

    // Thermal image: untouched, vertically centered in the right-hand panel.
    const int thermal_y_offset = ( canvas_height - cv_thermal_img.rows ) / 2;
    cv_thermal_img.copyTo( split_screen(
        cv::Rect( vis_img.cols, thermal_y_offset, cv_thermal_img.cols, cv_thermal_img.rows ) ) );

    vis_img = split_screen;
  }

  cv_bridge::CvImage cvImg;
  vis_img.copyTo( cvImg.image );
  // cvImg.header = img->header;
  cvImg.encoding = sensor_msgs::image_encodings::BGR8;
  // const auto info = std::make_shared<sensor_msgs::msg::CameraInfo>();
  // info->header = image->header;

  RCLCPP_DEBUG( get_logger(), "Publishing detection image" );

  auto vis_img_msg = cvImg.toImageMsg();
  vis_img_msg->header = cam_img->header;
  vis_pub_.publish( *vis_img_msg );
}

} // namespace hector_detection_aggregator
