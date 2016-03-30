#include "heat_detection.h"



int main(int argc, char **argv)
{

 // cv::namedWindow("Converted Image");

  ros::init(argc, argv, "heat_detection");

  ros::NodeHandle nh_("");
  ros::NodeHandle pnh_("~");
  HeatDetection hd(nh_, pnh_);

  ros::spin();

  return 0;
}

