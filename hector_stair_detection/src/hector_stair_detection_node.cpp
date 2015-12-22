#include <ros/ros.h>
#include <hector_stair_detection/hector_stair_detection.h>


int main(int argc, char **argv){
  ros::init(argc, argv, "hector_stair_detection_node");

  ROS_INFO("Starting HectorStairDetection Node");
  hector_stair_detection::HectorStairDetection obj;
  ros::spin();
  exit(0);
}
