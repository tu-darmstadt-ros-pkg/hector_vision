//
// Created by finn on 4/24/25.
//

#ifndef NEWEST_DATA_DETECTION_AGGREGATOR_HPP
#define NEWEST_DATA_DETECTION_AGGREGATOR_HPP

#include <cv_bridge/cv_bridge.hpp>

#include <hector_detection_aggregator/detection_aggregator_base.hpp>


namespace hector_detection_aggregator
{
/**
 * This aggregator is intended to provide the most up-to-date image feed possible.
 * All data is made available immediately after a new image is supplied, providing the image along with the most recent detections
 * Because of that detections are not synchronized to the image but appear delayed.
 */
class NewestDataDetectionAggregator : public DetectionAggregatorBase
{
private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Duration storage_duration_;
  cv_bridge::CvImageConstPtr current_image_;
  std::map<std::pair<std::string, std::string>, vision_msgs::msg::Detection2D> detection_map_;

public:
  /**
   * This aggregator is intended to provide the most up-to-date image feed possible.
   * All data is made available immediately after a new image is supplied, providing the image along with the most recent detections
   * Because of that detections are not synchronized to the image but appear delayed.
   * @param node The node is used to access the current ros-time
   * @param storage_duration How long each individual detection is stored and displayed after the last occurrence
   */
  NewestDataDetectionAggregator(const rclcpp::Node::SharedPtr &node,
    const rclcpp::Duration& storage_duration) : DetectionAggregatorBase(),
    node_(node), storage_duration_(storage_duration)
  {
    current_image_.reset();
  }

  /**
   * Add new camera image to the aggregator
   * @param image The newest camera image
   * @param camera_info Not used by this aggregator
   * @return Whether the collected data is ready to be displayed (always true)
   */
  bool AddImage(const sensor_msgs::msg::Image::ConstSharedPtr& image,
                const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info) override
  {
    current_image_ = cv_bridge::toCvShare( image, sensor_msgs::image_encodings::BGR8 );

    return true;
  }

  /**
   * Add new perception data to the aggregator
   * @param detections
   * @param info Not used by this aggregator
   * @return Whether the collected data is ready to be displayed (always false)
   */
  bool AddDetection(const vision_msgs::msg::Detection2DArray::ConstSharedPtr& detections,
                    const rclcpp::MessageInfo& info) override
  {
    for (const auto& detection : detections->detections)
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

    return false;
  }

  /**
   * Provides the previously supplied image along with the most recent occurrences of detections within the storage duration
   */
  std::pair<cv_bridge::CvImageConstPtr, std::vector<DetectionEntry>>
  GetAggregatedData() override
  {
    updateDetections();

    std::vector<DetectionEntry> detections;
    detections.reserve(detection_map_.size());
    for (const auto& [id, detection] : detection_map_)
    {
      detections.push_back({id.first, id.second, detection});
    }

    return {current_image_, detections};
  }

  /**
   * This aggregator does not consider different publishers
   * @param publisher_info Ignored
   */
  void UpdatePublishers(const std::vector<rclcpp::TopicEndpointInfo>& publisher_info) override {}

private:
  void updateDetections()
  {
    rclcpp::Time storage_threshold;
    if (node_->now().seconds() > storage_duration_.seconds())
      storage_threshold = node_->now() - storage_duration_;

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
};
}

#endif //NEWEST_DATA_DETECTION_AGGREGATOR_HPP
