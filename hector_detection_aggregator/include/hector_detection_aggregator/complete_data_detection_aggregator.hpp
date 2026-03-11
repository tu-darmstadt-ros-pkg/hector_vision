//
// Created by finn on 4/24/25.
//

#ifndef COMPLETE_DATA_DETECTION_AGGREGATOR_HPP
#define COMPLETE_DATA_DETECTION_AGGREGATOR_HPP

#include <hector_detection_aggregator/detection_aggregator_base.hpp>

namespace hector_detection_aggregator
{
struct ImageDetections {
  struct DetectorDetections {
    bool valid;
    std::vector<DetectionEntry> detections;

    DetectorDetections() : valid( false ) { }
  };

  std::vector<DetectorDetections> detectors_detections;

  explicit ImageDetections( const size_t detectors )
  {
    detectors_detections.reserve( detectors );
    for ( size_t i = 0; i < detectors; ++i ) detectors_detections.emplace_back();
  }

  bool DetectionsValid()
  {
    return std::all_of(
        detectors_detections.begin(), detectors_detections.end(),
        []( const DetectorDetections &detector_detections ) { return detector_detections.valid; } );
  }

  void InsertDetections( const size_t detector_index, const std::vector<DetectionEntry> &detections )
  {
    auto &[valid, detections_list] = detectors_detections.at( detector_index );
    detections_list = detections;
    valid = true;
  }

  std::vector<DetectionEntry> CollectDetections()
  {
    size_t total_detections = 0;

    for ( const auto &[_, detections] : detectors_detections )
      total_detections += detections.size();

    std::vector<DetectionEntry> all_detections;
    all_detections.reserve( total_detections );

    for ( const auto &[_, detections] : detectors_detections )
      all_detections.assign( detections.begin(), detections.end() );

    return all_detections;
  }
};

/**
 * This aggregator synchronizes received detections with the image frames they were detected on
 * so that every detection is displayed in the right place even if the view is moving quickly.
 * To guarantee that every detection is paired with the matching image this aggregator awaits
 * a detection from every detector before making it available.
 * This requires that all detectors consistently publish even if there are no detections as only
 * complete frames are ever published.
 * Depending on the delay of different detectors and the framerate of the camera a large buffer size
 * may be required for the aggregator to ever give results.
 */
class CompleteDataDetectionAggregator : public DetectionAggregatorBase
{
private:
  rclcpp::Node::SharedPtr node_;
  std::map<std::array<uint8_t, 16>, std::string> known_detectors_;
  std::map<std::array<uint8_t, 16>, int> detector_indexes_;
  size_t buffer_size_;
  size_t buffer_index_;
  size_t oldest_frame_index_;
  rclcpp::Time latest_valid_frame_;
  std::vector<rclcpp::Time> current_buffers_;
  std::map<rclcpp::Time, std::pair<cv_bridge::CvImageConstPtr, ImageDetections>> detections_;

public:
  /**
   * This aggregator synchronizes received detections with the image frames they were detected on
   * so that every detection is displayed in the right place even if the view is moving quickly.
   * To guarantee that every detection is paired with the matching image this aggregator awaits
   * a detection from every detector before making it available.
   * This requires that all detectors consistently publish even if there are no detections as only
   * complete frames are ever published.
   * Depending on the delay of different detectors and the framerate of the camera a large buffer
   * size may be required for the aggregator to ever give results.
   * @param node Used for logging
   * @param buffer_size How many frames are stored before the oldest is cleared
   * @param publisher_info Used to determine which detectors to expect
   */
  CompleteDataDetectionAggregator( const rclcpp::Node::SharedPtr &node, const size_t &buffer_size,
                                   const std::vector<rclcpp::TopicEndpointInfo> &publisher_info )
      : DetectionAggregatorBase(), node_( node ), buffer_size_( buffer_size ), buffer_index_( 0 ),
        oldest_frame_index_( 0 )
  {
    current_buffers_.reserve( buffer_size_ );
    for ( size_t i = 0; i < buffer_size_; ++i ) current_buffers_.push_back( node_->now() );
    detections_.clear();
    CompleteDataDetectionAggregator::UpdatePublishers( publisher_info );
  }

  bool AddImage( const sensor_msgs::msg::Image::ConstSharedPtr &image,
                 const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camera_info ) override
  {
    const rclcpp::Time image_time = { image->header.stamp };

    rclcpp::Time previous_time = current_buffers_.at( buffer_index_ );

    if ( image_time < previous_time ) {
      RCLCPP_WARN( node_->get_logger(), "Received image out of order. Image dropped." );
      return false;
    }

    const size_t new_index = buffer_index_ + 1 < buffer_size_ ? buffer_index_ + 1 : 0;

    if ( new_index == oldest_frame_index_ ) {
      RCLCPP_WARN( node_->get_logger(),
                   "A buffered frame was overwritten before it or a newer one could be published! "
                   "This may indicate that the buffer size is too small!" );
      detections_.erase( current_buffers_[oldest_frame_index_] );
      oldest_frame_index_ = oldest_frame_index_ + 1 < buffer_size_ ? oldest_frame_index_ + 1 : 0;
    }
    current_buffers_[new_index] = image_time;

    std::vector<DetectionEntry> detection_vector;
    detection_vector.reserve( known_detectors_.size() );
    detections_.emplace(
        image_time, std::pair( cv_bridge::toCvShare( image, sensor_msgs::image_encodings::BGR8 ),
                               ImageDetections( known_detectors_.size() ) ) );

    return false;
  }

  bool AddDetection( const vision_msgs::msg::Detection2DArray::ConstSharedPtr &detections,
                     const rclcpp::MessageInfo &info ) override
  {
    const rclcpp::Time detections_time = { detections->header.stamp };
    std::array<uint8_t, 16> gid_array{};
    const uint8_t *gid = info.get_rmw_message_info().publisher_gid.data;
    for ( size_t i = 0; i < 16; ++i ) gid_array[i] = gid[i];

    // Determine from which detector the detection came
    size_t detector_index;
    std::string detector_name;
    if ( const auto detector_entry = known_detectors_.find( gid_array );
         detector_entry != known_detectors_.end() ) {
      detector_name = detector_entry->second;
      detector_index = detector_indexes_.find( gid_array )->second;
    } else {
      RCLCPP_INFO( node_->get_logger(), "Ignoring detections from unknown detector" );
      return false;
    }

    // Find a matching image frame
    const auto entry = detections_.find( detections_time );
    if ( entry == detections_.end() ) {
      RCLCPP_DEBUG_STREAM( node_->get_logger(), "Could not insert detection from "
                                                    << detector_name
                                                    << " because no matching image was found!" );
      return false;
    }

    // Append detection to image frame
    std::vector<DetectionEntry> detection_vector;
    detection_vector.reserve( detections->detections.size() );
    for ( const auto &detection : detections->detections ) {
      vision_msgs::msg::ObjectHypothesis likeliest_type;
      likeliest_type.class_id = "nothing";
      likeliest_type.score = 0.0;

      for ( const auto &hypothesis : detection.results ) {
        const auto &type_hypothesis = hypothesis.hypothesis;
        if ( type_hypothesis.score > likeliest_type.score )
          likeliest_type = type_hypothesis;
      }

      if ( likeliest_type.class_id == "nothing" )
        continue;

      detection_vector.emplace_back( likeliest_type.class_id, detection.id, detection );
    }

    // Determine if the image frame is complete
    entry->second.second.InsertDetections( detector_index, detection_vector );
    if ( entry->second.second.DetectionsValid() ) {
      latest_valid_frame_ = entry->first;
      return true;
    }
    return false;
  }

  std::pair<cv_bridge::CvImageConstPtr, std::vector<DetectionEntry>> GetAggregatedData() override
  {
    const auto frame = detections_.find( latest_valid_frame_ );
    if ( frame == detections_.end() ) {
      RCLCPP_ERROR( node_->get_logger(), "No valid frame exists to display!" );
      throw std::exception();
    }

    std::pair data = { frame->second.first, frame->second.second.CollectDetections() };
    const rclcpp::Time frame_time = latest_valid_frame_;

    int frame_index = 0;
    for ( int i = 0; i < buffer_size_; ++i ) {
      if ( current_buffers_[i] == latest_valid_frame_ ) {
        frame_index = i;
        break;
      }
    }

    if ( frame_index == buffer_index_ ) {
      // Reset indexes if buffer is empty
      oldest_frame_index_ = 0;
      buffer_index_ = 0;
    } else
      oldest_frame_index_ = frame_index + 1 < buffer_size_ ? frame_index + 1 : 0;

    // Clear obsolete frames
    std::vector<rclcpp::Time> times_before_frame;
    times_before_frame.reserve( buffer_size_ );
    std::copy_if( current_buffers_.begin(), current_buffers_.end(), times_before_frame.begin(),
                  [frame_time]( const rclcpp::Time &t ) { return t >= frame_time; } );

    for ( const auto &obsolete_time : times_before_frame ) detections_.erase( obsolete_time );

    return data;
  }

  void UpdatePublishers( const std::vector<rclcpp::TopicEndpointInfo> &publisher_info ) override
  {
    std::map<std::array<uint8_t, 16>, std::string> current_publishers;

    // Assemble table of current detectors
    for ( const auto &publisher : publisher_info )
      current_publishers.emplace( publisher.endpoint_gid(), publisher.node_name() );

    // Determine if detectors have remained unchanged
    if ( known_detectors_ == current_publishers )
      return;

    // Find differences
    std::vector<std::string> lost_publishers;
    std::vector<std::string> new_publishers;
    for ( const auto &[gid, name] : known_detectors_ )
      if ( current_publishers.find( gid ) == current_publishers.end() )
        lost_publishers.push_back( name );
    for ( const auto &[gid, name] : current_publishers )
      if ( known_detectors_.find( gid ) == known_detectors_.end() )
        new_publishers.push_back( name );

    // Log differences
    std::stringstream info_log;
    info_log << "The detectors have changed:" << std::endl;
    if ( !lost_publishers.empty() ) {
      info_log << "Lost publishers: ";
      for ( const std::string &name : lost_publishers ) info_log << name << ", ";
      info_log << std::endl;
    }
    if ( !new_publishers.empty() ) {
      info_log << "Found new publishers: ";
      for ( const std::string &name : new_publishers ) info_log << name << ", ";
      info_log << std::endl;
    }
    RCLCPP_INFO_STREAM( node_->get_logger(), info_log.str() );

    // Apply changes
    known_detectors_ = current_publishers;
    detector_indexes_.clear();
    int i = 0;
    for ( const auto &[gid, _] : known_detectors_ ) {
      detector_indexes_.emplace( gid, i );
      ++i;
    }

    // Adapt buffer
    detections_.clear();
    buffer_index_ = 0;
  }
};
} // namespace hector_detection_aggregator

#endif // COMPLETE_DATA_DETECTION_AGGREGATOR_HPP
