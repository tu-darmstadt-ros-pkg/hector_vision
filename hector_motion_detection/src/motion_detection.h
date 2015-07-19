#ifndef _HECTOR_MOTION_DETECTION_H_
#define _HECTOR_MOTION_DETECTION_H_

#include <ros/ros.h>

#include <queue>

#include <opencv/cv.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <image_transport/image_transport.h>

#include <std_msgs/String.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CompressedImage.h>

#include <hector_worldmodel_msgs/ImagePercept.h>

#include <dynamic_reconfigure/server.h>
#include <hector_motion_detection/MotionDetectionConfig.h>

//#define DEBUG

using hector_motion_detection::MotionDetectionConfig;

class MotionDetection
{
public:
  MotionDetection();
  ~MotionDetection();

  typedef std::vector<cv::KeyPoint> KeyPoints;

private:
  void colorizeDepth(const cv::Mat& gray, cv::Mat& rgb) const;
  void drawOpticalFlowVectors(cv::Mat& img, const cv::Mat& optical_flow, int step = 10, const cv::Scalar& color = CV_RGB(0, 255, 0)) const;

  void computeOpticalFlow(const cv::Mat& prev_img, const cv::Mat& cur_img, cv::Mat& optical_flow, bool use_initial_flow = false, bool filter = false) const;
  void computeOpticalFlowMagnitude(const cv::Mat& optical_flow, cv::Mat& optical_flow_mag) const;

  void drawBlobs(cv::Mat& img, const KeyPoints& keypoints, double scale = 1.0) const;
  void detectBlobs(const cv::Mat& img, KeyPoints& keypoints) const;

  void update(const ros::TimerEvent& event);

  void imageCallback(const sensor_msgs::ImageConstPtr& img);

  void dynRecParamCallback(MotionDetectionConfig& config, uint32_t level);

  ros::Timer update_timer;

  image_transport::Subscriber image_sub_;

  ros::Publisher image_percept_pub_;
  image_transport::CameraSubscriber camera_sub_;
  image_transport::CameraPublisher image_motion_pub_;
  image_transport::CameraPublisher image_detected_pub_;

  dynamic_reconfigure::Server<MotionDetectionConfig> dyn_rec_server_;

  sensor_msgs::ImageConstPtr last_img;

  cv_bridge::CvImageConstPtr img_prev_ptr_;
  cv_bridge::CvImageConstPtr img_prev_col_ptr_;

  cv::Mat optical_flow;
  std::list<cv::Mat> flow_history;

  // dynamic reconfigure params
  double motion_detect_downscale_factor_;
  double motion_detect_inv_sensivity_;
  bool motion_detect_use_initial_flow_;
  bool motion_detect_image_flow_filter_;
  int motion_detect_threshold_;
  double motion_detect_min_area_;
  double motion_detect_min_blob_dist_;
  int motion_detect_dilation_size_;
  int motion_detect_flow_history_size_;
  std::string percept_class_id_;
};

#endif
