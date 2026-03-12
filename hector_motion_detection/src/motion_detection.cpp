// Author: Matej Zecevic
#include <hector_ros2_utils/node.hpp>
#include <rclcpp/rclcpp.hpp>

#include <image_transport/image_transport.hpp>
#include <std_msgs/msg/bool.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

#include <hector_motion_detection/motion_detection.hpp>

namespace hector_motion_detection
{

MotionDetection::MotionDetection( const rclcpp::NodeOptions &options )
    : Node( "motion_detection_node", options ), has_subscribers_( false ),
      first_image_received_( false )
{
  using namespace std::placeholders;
  bg_subtractor_ = cv::createBackgroundSubtractorMOG2();

  declare_reconfigurable_parameter(
      "moving_average_weight", std::ref<double>( moving_average_weight_ ),
      "Weight of the new image", hector::ParameterOptions<double>().setRange( 0.0, 1.0, 0.01 ) );
  declare_reconfigurable_parameter( "motion_detect_activation_threshold",
                                    std::ref<int>( activation_threshold_ ),
                                    "Threshold for a pixel to be considered 'moving'",
                                    hector::ParameterOptions<int>().setRange( 0, 255, 1 ) );
  declare_reconfigurable_parameter( "motion_detect_automatic_learning_rate",
                                    std::ref<bool>( automatic_learning_rate_ ),
                                    "Automatic learning rate", hector::ParameterOptions<bool>() );
  declare_reconfigurable_parameter( "motion_detect_learning_rate", std::ref<double>( learning_rate_ ),
                                    "Learning rate for background subtraction",
                                    hector::ParameterOptions<double>().setRange( 0.7, 1.0, 0.0 ) );
  declare_reconfigurable_parameter( "motion_detect_detection_limit", std::ref<int>( detectionLimit_ ),
                                    "Maximum number of motions to detect",
                                    hector::ParameterOptions<int>().setRange( 0, 8, 1 ) );
  declare_reconfigurable_parameter( "motion_detect_min_area", std::ref<int>( min_area_ ),
                                    "Minimal area of detected motions",
                                    hector::ParameterOptions<int>().setRange( 0, 10000, 1 ) );
  declare_reconfigurable_parameter( "motion_detect_max_area", std::ref<int>( max_area_ ),
                                    "Maximal area of detected motions",
                                    hector::ParameterOptions<int>().setRange( 5000, 20000, 1 ) );
  declare_reconfigurable_parameter( "motion_detect_erosion", std::ref<int>( erosion_iterations_ ),
                                    "Iterations for erosion on fgimg",
                                    hector::ParameterOptions<int>().setRange( 0, 8, 1 ) );
  declare_reconfigurable_parameter( "motion_detect_dilation", std::ref<int>( dilation_iterations_ ),
                                    "Iterations for dilation on fgimg",
                                    hector::ParameterOptions<int>().setRange( 0, 16, 1 ) );
  declare_reconfigurable_parameter( "motion_detect_shadows", std::ref<bool>( shadows_ ),
                                    "Whether shadows should be tracked",
                                    hector::ParameterOptions<bool>() );
  declare_reconfigurable_parameter( "motion_detect_debug_contours", std::ref<bool>( debug_contours_ ),
                                    "For tracking the contours", hector::ParameterOptions<bool>() );
  declare_reconfigurable_parameter(
      "motion_detect_debug_images", std::ref<bool>( debug_images_ ),
      "Whether to advertise debug image topics",
      hector::ParameterOptions<bool>().onUpdate(
          [this]( const bool &enabled ) { debugPublisherCallback( enabled ); } ) );
  // Parameter for image transport
  declare_parameter<std::string>( "image_transport", "raw" );

  RCLCPP_INFO( get_logger(), "Starting Motion Detection with MOG2" );
  RCLCPP_INFO( get_logger(), "debug_contours: %i", debug_contours_ );
  RCLCPP_INFO( get_logger(), "shadows: %i", shadows_ );
  RCLCPP_INFO( get_logger(), "max area: %d", max_area_ );
  RCLCPP_INFO( get_logger(), "min area: %d", min_area_ );
  RCLCPP_INFO( get_logger(), "detection limit: %d", detectionLimit_ );

  // image_node_ = rclcpp::Node::make_shared("motion_detection_image_node", options);
  image_transport_ = std::make_shared<image_transport::ImageTransport>( shared_from_this() );

  debugPublisherCallback( debug_images_ );

  image_perception_pub_ =
      create_publisher<vision_msgs::msg::Detection2DArray>( "detection/image_detection", 10 );

  using std::chrono_literals::operator""s;
  check_subscriptions_timer_ =
      create_wall_timer( 1s, std::bind( &MotionDetection::publisherSubscriptionCallback, this ) );
}

void MotionDetection::imageCallback( const sensor_msgs::msg::Image::ConstSharedPtr &img )
{
  RCLCPP_DEBUG( get_logger(), "Received image on topic %s", image_sub_.getTopic().c_str() );
  cv_bridge::CvImageConstPtr cv_ptr;
  cv_ptr = cv_bridge::toCvShare( img, sensor_msgs::image_encodings::BGR8 );
  cv::Mat frame( cv_ptr->image );

  cv::Mat fgimg;
  if ( automatic_learning_rate_ ) {
    bg_subtractor_->apply( frame, fgimg );
  } else {
    bg_subtractor_->apply( frame, fgimg, learning_rate_ );
  }

  cv::Mat fgimg_orig;
  fgimg.copyTo( fgimg_orig ); // For debugging/tuning purposes

  cv::morphologyEx( fgimg, fgimg, cv::MORPH_CLOSE,
                    cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 3, 3 ) ) );

  // Controllable iterations for morphological operations
  for ( int i = 0; i < erosion_iterations_; i++ ) { cv::erode( fgimg, fgimg, cv::Mat() ); }
  for ( int i = 0; i < dilation_iterations_; i++ ) {
    cv::dilate( fgimg, fgimg, cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 3, 3 ) ) );
    // Alternatively (tuning): other kernels: e.g. cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10,10))
  }

  // Moving average of fg image
  if ( !first_image_received_ ) {
    fgimg.copyTo( accumulated_image_ );
    first_image_received_ = true;
  } else {
    cv::addWeighted( accumulated_image_, ( 1 - moving_average_weight_ ), fgimg,
                     moving_average_weight_, 0.0, accumulated_image_ );
  }
  cv::Mat thresholded;
  cv::threshold( accumulated_image_, thresholded, activation_threshold_, 255, cv::THRESH_BINARY );

  // Find contours
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours( thresholded, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
  if ( debug_contours_ ) {
    cv::drawContours( frame, contours, -1, cv::Scalar( 0, 0, 255 ), 2 );
  }

  // ======== Detection of g largest contours =====
  int largest_area = 0;
  int largest_contour_index = 0;
  cv::Rect bounding_rect;
  std::vector<double> areas( contours.size() );

  for ( size_t i = 0; i < contours.size(); i++ ) {
    double area = cv::contourArea( contours[i] ); //  Find the area of contour
    areas[i] = area;
  }

  std::vector<vision_msgs::msg::Detection2D> polygonGroup;
  vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
  hypothesis.hypothesis.class_id = "motion";
  hypothesis.hypothesis.score = 1.0;

  if ( contours.size() != 0 ) {
    for ( int k = 0; k < detectionLimit_; k++ ) {
      for ( size_t j = 0; j < areas.size(); j++ ) {
        if ( areas[j] > largest_area ) {
          largest_area = areas[j];
          // Store the index of largest contour
          largest_contour_index = j;
          // Find the bounding rectangle for biggest contour
          bounding_rect = cv::boundingRect( contours[j] );
        }
      }
      if ( areas[largest_contour_index] >= min_area_ && areas[largest_contour_index] <= max_area_ ) {
        cv::rectangle( frame, bounding_rect.tl(), bounding_rect.br(), cv::Scalar( 0, 0, 255 ), 2, 8,
                       0 );

        vision_msgs::msg::Detection2D perceptionData;
        perceptionData.id = "motion_" + std::to_string( k );
        perceptionData.header = img->header;
        perceptionData.results.push_back( hypothesis );

        perceptionData.bbox.center.position.x = ( bounding_rect.tl().x + bounding_rect.br().x ) / 2;
        perceptionData.bbox.center.position.y = ( bounding_rect.tl().y + bounding_rect.tl().y ) / 2;
        perceptionData.bbox.size_x = bounding_rect.width;
        perceptionData.bbox.size_y = bounding_rect.height;

        polygonGroup.push_back( perceptionData );
      }
      areas[largest_contour_index] = -1;
      largest_area = 0;
    }
    if ( image_perception_pub_->get_subscription_count() > 0 ) {
      vision_msgs::msg::Detection2DArray polygonPerceptionArray;
      polygonPerceptionArray.header.stamp = img->header.stamp;
      polygonPerceptionArray.detections = polygonGroup;
      image_perception_pub_->publish( polygonPerceptionArray );
    }
  }
  sensor_msgs::msg::CameraInfo::SharedPtr info;
  info.reset( new sensor_msgs::msg::CameraInfo() );
  info->header = img->header;

  if ( debug_images_ ) {
    if ( image_background_subtracted_pub_.getNumSubscribers() > 0 ) {
      cv_bridge::CvImage cvImg;
      cvImg.image = fgimg_orig;
      cvImg.header = img->header;
      cvImg.encoding = sensor_msgs::image_encodings::MONO8;
      image_background_subtracted_pub_.publish( cvImg.toImageMsg(), info );
    }

    if ( image_motion_pub_.getNumSubscribers() > 0 ) {
      cv_bridge::CvImage cvImg;
      cvImg.image = thresholded;
      cvImg.header = img->header;
      cvImg.encoding = sensor_msgs::image_encodings::MONO8;
      image_motion_pub_.publish( cvImg.toImageMsg(), info );
    }

    if ( image_detected_pub_.getNumSubscribers() > 0 ) {
      cv_bridge::CvImage cvImg;
      cvImg.image = frame;
      cvImg.header = img->header;
      cvImg.encoding = sensor_msgs::image_encodings::BGR8;
      image_detected_pub_.publish( cvImg.toImageMsg(), info );
    }
  }
}

void MotionDetection::debugPublisherCallback( const bool &enabled )
{
  if ( enabled ) {
    RCLCPP_INFO( get_logger(), "Enabling debug publishers" );

    image_motion_pub_ = image_transport_->advertiseCamera( "image_motion", 10 );
    image_detected_pub_ = image_transport_->advertiseCamera( "image_detected", 10 );
    image_background_subtracted_pub_ =
        image_transport_->advertiseCamera( "image_background_subtracted", 10 );
  } else {
    RCLCPP_INFO( get_logger(), "Disabling debug publishers" );

    image_motion_pub_.shutdown();
    image_detected_pub_.shutdown();
    image_background_subtracted_pub_.shutdown();
  }
}

void MotionDetection::publisherSubscriptionCallback()
{
  size_t subscribers = image_perception_pub_->get_subscription_count();
  if ( debug_images_ )
    subscribers += image_motion_pub_.getNumSubscribers() + image_detected_pub_.getNumSubscribers() +
                   image_background_subtracted_pub_.getNumSubscribers();

  RCLCPP_DEBUG( get_logger(), "Node has %3lu subscriber%s and previously %s subscribers",
                subscribers, subscribers == 1 ? "" : "s", has_subscribers_ ? "had" : "didn't have" );

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

void MotionDetection::startSubscribers()
{
  RCLCPP_INFO( get_logger(), "Starting subscriber" );
  image_sub_ = image_transport_->subscribe( "image", 10, &MotionDetection::imageCallback, this );
}

void MotionDetection::stopSubscribers()
{
  RCLCPP_INFO( get_logger(), "Stopping subscriber" );
  image_sub_.shutdown();
}

} // namespace hector_motion_detection

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE( hector_motion_detection::MotionDetection );
