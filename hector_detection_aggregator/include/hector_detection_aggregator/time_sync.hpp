
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>

#include <hector_ros2_utils/node.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>

#include <hector_perception_msgs/msg/object_detection2_d_array.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include <hector_detection_aggregator/hector_detection_aggregator_params.hpp>

namespace hector_detection_aggregator
{

template<typename StorageType>
class FixedSizeStampedBuffer
{
public:
  FixedSizeStampedBuffer( std::size_t size )
  {
    buffer_ = std::map<int64_t, std::shared_ptr<const StorageType>>();
    size_ = size;
  }

  void insert( const int64_t stamp, std::shared_ptr<const StorageType> data )
  {
    std::lock_guard guard( buffer_mutex );
    if ( buffer_.size() == size_ )
      // Erase oldest element
      buffer_.erase( buffer_.begin() );

    buffer_.emplace( stamp, data );
  };

  virtual typename std::map<int64_t, std::shared_ptr<const StorageType>>::iterator
  find_matching_entry( const int64_t stamp ) = 0;

  virtual typename std::shared_ptr<const StorageType> find( const int64_t stamp )
  {
    std::lock_guard guard( buffer_mutex );

    const auto it = find_matching_entry( stamp );
    if ( it == buffer_.end() )
      return nullptr;

    return it->second;
  };

  // Call if element was sucessfully processed and can be erased permanently
  virtual void erase( const int64_t stamp )
  {
    std::lock_guard guard( buffer_mutex );
    buffer_.erase( stamp );
  };

protected:
  std::size_t size_;
  std::map<int64_t, std::shared_ptr<const StorageType>> buffer_;
  std::mutex buffer_mutex;
};

// Dummies buffers return a default <Storagetype> object for any time stamp query.
// This allows a determination of actually active inputs at runtime since callback signatures
// and completeness checks need to be defined at compile time.
// The already available message filter api struggles with this since absent inputs would stop the
// whole filter from ever completing.
template<typename StorageType>
class DummyFixedSizeStampedBuffer : public FixedSizeStampedBuffer<StorageType>
{
public:
  DummyFixedSizeStampedBuffer()
      : FixedSizeStampedBuffer<StorageType>( 0 ), dummy_( std::make_shared<StorageType>() ) { };

private:
  typename std::map<int64_t, std::shared_ptr<const StorageType>>::iterator
  find_matching_entry( const int64_t ) override
  { return this->buffer_.begin(); }; // Unused in this class
  typename std::shared_ptr<const StorageType> find( const int64_t ) override { return dummy_; };
  void erase( const int64_t ) override { };

  const std::shared_ptr<const StorageType> dummy_;
};

template<typename StorageType>
class ExactTimeBuffer : public FixedSizeStampedBuffer<StorageType>
{
public:
  ExactTimeBuffer( std::size_t size ) : FixedSizeStampedBuffer<StorageType>( size ) { }

private:
  virtual typename std::map<int64_t, std::shared_ptr<const StorageType>>::iterator
  find_matching_entry( const int64_t stamp )
  { return this->buffer_.find( stamp ); };
};

using Detection2DArray = hector_perception_msgs::msg::ObjectDetection2DArray;
// Signature cam image, thermal image, motion, object detection (april/QR/hazmat)
using DistributionCallback = std::function<void(
    std::shared_ptr<const sensor_msgs::msg::Image>, std::shared_ptr<const sensor_msgs::msg::Image>,
    std::shared_ptr<const Detection2DArray>, std::shared_ptr<const Detection2DArray> )>;

class TimeSyncFilter
{
public:
  TimeSyncFilter() { };

  void init( std::shared_ptr<rclcpp::Node> node, DistributionCallback dataCb,
             std::shared_ptr<detection_aggregator::Params> params,
             std::shared_ptr<image_transport::ImageTransport> image_transport );

private:
  template<typename T>
  void insertAndCheck( std::shared_ptr<FixedSizeStampedBuffer<T>> buffer,
                       const std::shared_ptr<const T> msg );

  void DistributeIfComplete( int64_t stamp );

  std::shared_ptr<image_transport::Subscriber> SetupImageTransport(
      const std::string &topic,
      const std::function<void( const std::shared_ptr<const sensor_msgs::msg::Image> & )> &fp,
      const rclcpp::SubscriptionOptions &options );

  std::shared_ptr<rclcpp::Node> node_;
  std::shared_ptr<image_transport::ImageTransport> image_transport_;
  std::shared_ptr<detection_aggregator::Params> params_;

  bool use_thermal_;
  bool use_motion_;
  bool use_obj_detection_;

  DistributionCallback dataCb_;

  std::shared_ptr<image_transport::Subscriber> cam_sub_;
  std::shared_ptr<image_transport::Subscriber> thermal_sub_;
  rclcpp::Subscription<Detection2DArray>::SharedPtr motion_sub_;
  rclcpp::Subscription<Detection2DArray>::SharedPtr object_detection_sub_;

  std::shared_ptr<FixedSizeStampedBuffer<sensor_msgs::msg::Image>> cam_buffer_;
  std::shared_ptr<FixedSizeStampedBuffer<sensor_msgs::msg::Image>> thermal_buffer_;
  std::shared_ptr<FixedSizeStampedBuffer<Detection2DArray>> motion_buffer_;
  std::shared_ptr<FixedSizeStampedBuffer<Detection2DArray>> obj_detection_buffer_;
};

} // namespace hector_detection_aggregator