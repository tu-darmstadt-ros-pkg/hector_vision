#ifndef _HECTOR_SOFT_OBSTACLE_DETECTION_H_
#define _HECTOR_SOFT_OBSTACLE_DETECTION_H_

#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <opencv/cv.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

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
  void transformScanToImage(const sensor_msgs::LaserScanConstPtr scan, cv::Mat& img, cv::Point2i& scan_center) const;
  void houghTransform(const cv::Mat& img, std::vector<cv::Vec4i>& lines) const;
  void getLine(const cv::Mat& img, const cv::Vec4i& line, cv::Mat& out) const;
  void evalModel(const std::vector<double>& edges, const std::vector<double>& centers, double beta, double& r, double& dr, double& Hr) const;
  double computeFrequency(const cv::Mat& signal, double& mean, double& var) const;

  void lineToPointCloud(const cv::Vec4i& line, const cv::Point2i& scan_center, pcl::PointCloud<pcl::PointXYZ>& out) const;

  void update(const ros::TimerEvent& event);

  void laserScanCallback(const sensor_msgs::LaserScanConstPtr& scan);

  void dynRecParamCallback(SoftObstacleDetectionConfig& config, uint32_t level);

  ros::Timer update_timer;

  sensor_msgs::LaserScanConstPtr last_scan;

  double unit_scale;
  int border_size;

  tf::TransformListener tf_listener;

  // subscriber
  ros::Subscriber laser_scan_sub_;

  // publisher
  ros::Publisher veil_percept_pub_;
  ros::Publisher veil_percept_pose_pub_;
  ros::Publisher veil_point_cloud_pub_;

  dynamic_reconfigure::Server<SoftObstacleDetectionConfig> dyn_rec_server_;

  // dynamic reconfigure params
  double max_curtain_length_sq_;
  double min_frequency_;
  double max_frequency_;
  double max_var_;
  std::string percept_class_id_;
};

#endif
