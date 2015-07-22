#ifndef _HECTOR_SOFT_OBSTACLE_DETECTION_H_
#define _HECTOR_SOFT_OBSTACLE_DETECTION_H_

#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

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
  uchar getMaxAtLine(const cv::Mat& img, const cv::Point& p1, const cv::Point& p2) const;
  void getLine(const cv::Mat& img, const cv::Vec4i& line, cv::Mat& out) const;
  void edgeDetection(const cv::Mat& signal, std::vector<double>& edges, std::vector<double>& centers) const;

  bool checkSegmentsMatching(const std::vector<double>& edges, const std::vector<double>& centers, double veil_segment_size, double min_segments, double max_segments) const;

  double evalModel(const std::vector<double>& edges, const std::vector<double>& centers, double lambda) const;
  bool checkFrequencyMatching(const std::vector<double>& edges, const std::vector<double>& centers, double lambda, double min_segments, double max_segments) const;

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
  unsigned int min_hole_size_;
  double max_curtain_length_sq_;
  double min_frequency_;
  double max_frequency_;
  double veil_segment_size_;
  int min_segments_;
  int max_segments_;
  double max_segment_size_mse_;
  double max_segment_size_var_;
  double max_segment_dist_var_;
  double size_dist_ratio_;
  double max_frequency_mse_;
  std::string percept_class_id_;
};

#endif
