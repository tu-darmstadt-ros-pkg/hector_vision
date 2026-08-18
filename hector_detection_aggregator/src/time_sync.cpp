#include "hector_detection_aggregator/time_sync.hpp"

namespace hector_detection_aggregator
{

std::shared_ptr<image_transport::Subscriber> TimeSyncFilter::setupImageTransport(
    const std::string &topic,
    const std::function<void( const std::shared_ptr<const sensor_msgs::msg::Image> & )> &fp,
    const rclcpp::SubscriptionOptions &options )
{
  std::string base_topic = topic;
  std::string transport = "raw";

  const std::string suffix = "/compressed";
  if ( topic.size() > suffix.size() &&
       topic.compare( topic.size() - suffix.size(), suffix.size(), suffix ) == 0 ) {
    base_topic = topic.substr( 0, topic.size() - suffix.size() );
    transport = "compressed";
  }

  auto transport_hint = image_transport::TransportHints( node_.get(), transport );

  return std::make_shared<image_transport::Subscriber>(
      image_transport_->subscribe( base_topic, 10, fp,
                                   image_transport::ImageTransport::VoidPtr(), // tracked_object
                                   &transport_hint,                            // transport_hints
                                   options ) );
}

void TimeSyncFilter::init( std::shared_ptr<rclcpp::Node> node, DistributionCallback dataCb,
                           std::shared_ptr<detection_aggregator::Params> params,
                           std::shared_ptr<image_transport::ImageTransport> image_transport )
{
  node_ = node;
  dataCb_ = dataCb;

  image_transport_ = image_transport;

  params_ = params;
  use_thermal_ = params_->use_thermal;
  use_motion_ = params_->use_motion;
  use_obj_detection_ = ( params->use_hazmat || params->use_april || params->use_qr );
  // Make detection topics run in parallel to increase throughput. Timebuffers are designed threadsafe by utilizing mutexes.
  auto reentrant_cb_group = node_->create_callback_group( rclcpp::CallbackGroupType::Reentrant );
  rclcpp::SubscriptionOptions options;
  options.callback_group = reentrant_cb_group;

  RCLCPP_INFO( node_->get_logger(), "Setting up image transport for %s", params->cam_topic.c_str() );

  std::lock_guard distrib_guard(
      distribution_mutex_ ); // Prevents checking of any buffers before they are initialized

  cam_buffer_ = std::make_shared<ExactTimeBuffer<sensor_msgs::msg::Image>>( params->buffer_size );
  cam_sub_ = setupImageTransport( params->cam_topic,
                                  std::bind( &TimeSyncFilter::insertAndCheck<sensor_msgs::msg::Image>,
                                             this, cam_buffer_, std::placeholders::_1 ),
                                  options );

  if ( use_thermal_ ) {
    thermal_buffer_ = std::make_shared<ApproximateTimeBuffer<sensor_msgs::msg::Image>>(
        params->buffer_size, 1e9 ); // 1s tolerance
    thermal_sub_ = setupImageTransport(
        params->thermal_topic,
        [&]( const std::shared_ptr<const sensor_msgs::msg::Image> &msg ) {
          insertAndCheck<sensor_msgs::msg::Image>( thermal_buffer_, msg );
        },
        options );
  } else
    thermal_buffer_ = std::make_shared<DummyFixedSizeStampedBuffer<sensor_msgs::msg::Image>>();

  if ( use_motion_ ) {
    motion_buffer_ = std::make_shared<ExactTimeBuffer<Detection2DArray>>( params->buffer_size );
    motion_sub_ = node_->create_subscription<Detection2DArray>(
        params->motion_topic, rclcpp::SensorDataQoS(),
        [&]( const std::shared_ptr<const hector_perception_msgs::msg::ObjectDetection2DArray> &msg ) {
          insertAndCheck<hector_perception_msgs::msg::ObjectDetection2DArray>( motion_buffer_, msg );
        },
        options );
  } else
    motion_buffer_ = std::make_shared<DummyFixedSizeStampedBuffer<Detection2DArray>>();
  if ( use_obj_detection_ ) {
    obj_detection_buffer_ =
        std::make_shared<ExactTimeBuffer<Detection2DArray>>( params->buffer_size );
    object_detection_sub_ = node_->create_subscription<Detection2DArray>(
        params->object_detection_topic, rclcpp::SensorDataQoS(),
        [&]( const std::shared_ptr<const hector_perception_msgs::msg::ObjectDetection2DArray> &msg ) {
          insertAndCheck<hector_perception_msgs::msg::ObjectDetection2DArray>( obj_detection_buffer_,
                                                                               msg );
        },
        options );
  } else
    obj_detection_buffer_ = std::make_shared<DummyFixedSizeStampedBuffer<Detection2DArray>>();
}

void TimeSyncFilter::distributeIfComplete( int64_t stamp )
{
  distribution_mutex_.lock();

  auto m1 = cam_buffer_->find( stamp );
  auto m2 = thermal_buffer_->find( stamp );
  auto m3 = motion_buffer_->find( stamp );
  auto m4 = obj_detection_buffer_->find( stamp );

  if ( m1 && m2 && m3 && m4 ) {
    cam_buffer_->erase( stamp );
    thermal_buffer_->erase( stamp );
    motion_buffer_->erase( stamp );
    obj_detection_buffer_->erase( stamp );

    distribution_mutex_
        .unlock(); // Unlocking safe here since data has been erased which prevents repetitive data processing

    dataCb_( m1, m2, m3, m4 );
  }
}

template<typename T>
void TimeSyncFilter::insertAndCheck( std::shared_ptr<FixedSizeStampedBuffer<T>> buffer,
                                     const std::shared_ptr<const T> msg )
{
  int64_t stamp = rclcpp::Time( msg->header.stamp ).nanoseconds();
  buffer->insert( stamp, msg );
  distributeIfComplete( stamp );
}

} // namespace hector_detection_aggregator