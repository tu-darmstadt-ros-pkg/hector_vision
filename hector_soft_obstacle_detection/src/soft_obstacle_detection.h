#ifndef _HECTOR_SOFT_OBSTACLE_DETECTION_H_
#define _HECTOR_SOFT_OBSTACLE_DETECTION_H_

#include <ros/ros.h>

#include <opencv/cv.h>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/LaserScan.h>

#include <dynamic_reconfigure/server.h>
#include <hector_soft_obstacle_detection/SoftObstacleDetectionConfig.h>

//#define DEBUG

using hector_soft_obstacle_detection::SoftObstacleDetectionConfig;

class SoftObstacleDetection
{
public:
  SoftObstacleDetection();
  ~SoftObstacleDetection();

private:
  void update(const ros::TimerEvent& event);

  void laserScanCallback(const sensor_msgs::LaserScanConstPtr& scan);

  void dynRecParamCallback(SoftObstacleDetectionConfig& config, uint32_t level);

  ros::Timer update_timer;

  sensor_msgs::LaserScanConstPtr last_scan;

  // subscriber
  ros::Subscriber laser_scan_sub_;

  dynamic_reconfigure::Server<SoftObstacleDetectionConfig> dyn_rec_server_;

  // dynamic reconfigure params
  std::string percept_class_id_;
};

#endif
