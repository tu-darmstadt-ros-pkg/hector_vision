#ifndef HECTOR_DETECTION_AGGREGATOR_HPP_
#define HECTOR_DETECTION_AGGREGATOR_HPP_

#include <map>
#include <memory>

#include <cv_bridge/cv_bridge.hpp>
#include <hector_ros2_utils/node.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <hector_perception_msgs/msg/object_detection2_d_array.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include <hector_detection_aggregator/hector_detection_aggregator_params.hpp>
#include <hector_detection_aggregator/time_sync.hpp>

using Detection2DArray = hector_perception_msgs::msg::ObjectDetection2DArray;
using Detection2D = hector_perception_msgs::msg::ObjectDetection2D;

namespace hector_detection_aggregator
{

class DetectionAggregator : public hector::Node
{
public:
  DetectionAggregator();
  bool setup();

private:
  std::vector<std::string> detection_topics_;

  std::shared_ptr<detection_aggregator::ParamListener> param_listener_;
  std::shared_ptr<detection_aggregator::Params> params_;

  std::shared_ptr<image_transport::ImageTransport> image_transport_;

  // std::map<std::pair<std::string, std::string>, vision_msgs::msg::Detection2D> detection_map_;
  std::map<std::string, cv::Scalar> color_map_;

  image_transport::Publisher vis_pub_;

  image_transport::Subscriber image_subscriber_;

  TimeSyncFilter time_sync_filter_;

  bool readParameters();

  void processDetectionSet( std::shared_ptr<const sensor_msgs::msg::Image> cam_img,
                            std::shared_ptr<const sensor_msgs::msg::Image> thermal_img,
                            std::shared_ptr<const Detection2DArray> motion_detection,
                            std::shared_ptr<const Detection2DArray> obj_detection );
};
} // namespace hector_detection_aggregator
#endif
