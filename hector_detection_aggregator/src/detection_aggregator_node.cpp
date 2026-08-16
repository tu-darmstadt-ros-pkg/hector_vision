#include "hector_detection_aggregator/detection_aggregator.hpp"
#include <rclcpp/rclcpp.hpp>

int main( int argc, char **argv )
{
  rclcpp::init( argc, argv );

  auto node = std::make_shared<hector_detection_aggregator::DetectionAggregator>();
  node->Setup();
  rclcpp::spin( node );
  rclcpp::shutdown();
  return 0;
}
