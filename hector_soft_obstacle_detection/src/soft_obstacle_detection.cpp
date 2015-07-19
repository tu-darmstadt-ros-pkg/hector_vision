#include "soft_obstacle_detection.h"

SoftObstacleDetection::SoftObstacleDetection()
{
  ros::NodeHandle nh;

  // subscribe topics
  laser_scan_sub_ = nh.subscribe("/laser1/scan", 1, &SoftObstacleDetection::laserScanCallback, this);

  // advertise topics
//  image_percept_pub_ = nh.advertise<hector_worldmodel_msgs::ImagePercept>("image_percept", 20);
//  image_motion_pub_ = image_motion_it.advertiseCamera("image_motion", 10);
//  image_detected_pub_ = image_detected_it.advertiseCamera("image_detected", 10);

  // dynamic reconfigure
  dyn_rec_server_.setCallback(boost::bind(&SoftObstacleDetection::dynRecParamCallback, this, _1, _2));

  update_timer = nh.createTimer(ros::Duration(0.04), &SoftObstacleDetection::update, this);
}

SoftObstacleDetection::~SoftObstacleDetection()
{
}

void SoftObstacleDetection::update(const ros::TimerEvent& /*event*/)
{
  if (!last_scan)

  last_scan.reset();
}

void SoftObstacleDetection::laserScanCallback(const sensor_msgs::LaserScanConstPtr& scan)
{
  last_scan = scan;
}

void SoftObstacleDetection::dynRecParamCallback(SoftObstacleDetectionConfig& config, uint32_t level)
{
  percept_class_id_ = config.percept_class_id;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "soft_obstacle_detection");
  SoftObstacleDetection sod;
  ros::spin();

  return 0;
}

