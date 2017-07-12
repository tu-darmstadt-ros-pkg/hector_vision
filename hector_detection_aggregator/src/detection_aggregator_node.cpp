#include <ros/ros.h>
#include <hector_detection_aggregator/detection_aggregator.h>

int main(int argc, char **argv){
  ros::init(argc, argv, "hector_detection_aggregator_node");
  hector_detection_aggregator::DetectionAggregator detection_aggregator;
  ROS_INFO("Starting HectorDectionAggregatorNode");
  ros::Rate rate(10);
  while (ros::ok())
  {
      detection_aggregator.createImage();
      rate.sleep();
      ros::spinOnce();
  }
  exit(0);
}
