//Author: Matej Zecevic
#include <hector_motion_detection/motion_detection.h>

namespace hector_motion_detection {

MotionDetection::MotionDetection(ros::NodeHandle &nh)
  : nh_(nh), first_image_received_(false)
{
    bg_subtractor_ = cv::createBackgroundSubtractorMOG2();

    ros::NodeHandle pnh("~"); //private nh

    pnh.param("enabled", enabled_, false);

    image_transport::ImageTransport it(nh);

    image_sub_ = it.subscribe("image", 1 , &MotionDetection::imageCallback, this);
    enabled_sub_ = nh.subscribe("enabled", 10, &MotionDetection::enabledCallback, this);
    enabled_pub_ = nh.advertise<std_msgs::Bool>("enabled_status", 10, true);

    publishEnableStatus();

    dyn_rec_type_ = boost::bind(&MotionDetection::dynRecParamCallback, this, _1, _2);
    dyn_rec_server_.setCallback(dyn_rec_type_);

    image_percept_pub_ = nh.advertise<hector_worldmodel_msgs::ImagePercept>("image_percept", 20);
    image_motion_pub_ = it.advertiseCamera("image_motion", 10);
    image_detected_pub_ = it.advertiseCamera("image_detected", 10);

    image_perception_pub = nh.advertise<hector_perception_msgs::PerceptionDataArray>("detection/image_detection", 10);
    ROS_INFO("Starting Motion Detection with MOG2");
    ROS_INFO("debug_contours: %i", debug_contours_);
    ROS_INFO("shadows: %i", shadows_);
    ROS_INFO("max area: %d", max_area_);
    ROS_INFO("min area: %d", min_area_);
    ROS_INFO("detection limit: %d", detectionLimit_);
    image_transport::ImageTransport image_bg_it(pnh);
    image_background_subtracted_pub_ = image_bg_it.advertiseCamera("image_background_subtracted", 10);
}

void MotionDetection::enabledCallback(const std_msgs::BoolConstPtr& enabled) {
  enabled_ = enabled->data;
  publishEnableStatus();
}

void MotionDetection::imageCallback(const sensor_msgs::ImageConstPtr& img)
{
  if (!enabled_) {
    return;
  }
  cv_bridge::CvImageConstPtr cv_ptr;
  cv_ptr = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::BGR8);
  cv::Mat frame(cv_ptr->image);

  cv::Mat fgimg;
  if (automatic_learning_rate_) {
    bg_subtractor_->apply(frame, fgimg);
  } else {
    bg_subtractor_->apply(frame, fgimg, learning_rate_);
  }

  cv::Mat fgimg_orig;
  fgimg.copyTo(fgimg_orig);   //for debugging/tuning purposes

  cv::morphologyEx(fgimg, fgimg, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));

  //controlable iterations for morphological operations
  for(int i=0; i < erosion_iterations_; i++) {
    cv::erode(fgimg, fgimg, cv::Mat());
  }
  for(int i=0; i < dilation_iterations_; i++) {
    cv::dilate(fgimg, fgimg, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));
    //alternatively (tuning): other kernels: e.g. cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10,10))
  }

  // moving average of fg image
  if (!first_image_received_) {
    fgimg.copyTo(accumulated_image_);
    first_image_received_ = true;
  } else {
    cv::addWeighted(accumulated_image_, (1 - moving_average_weight_), fgimg, moving_average_weight_, 0.0, accumulated_image_);
  }
  cv::Mat thresholded;
  cv::threshold(accumulated_image_, thresholded, activation_threshold_, 255, cv::THRESH_BINARY);

  // Find contours
  std::vector<std::vector<cv::Point> > contours;
  cv::findContours (thresholded, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  if (debug_contours_) {
    cv::drawContours(frame, contours, -1, cv::Scalar (0, 0, 255), 2);
  }

  //========Detection of g largest contours=====
  int largest_area=0;
  int largest_contour_index=0;
  cv::Rect bounding_rect;
  std::vector<double> areas( contours.size() );

  for( int i = 0; i < contours.size(); i++ ) {
    double area = cv::contourArea( contours[i] );  //  Find the area of contour
    areas[i] = area;
  }

  std::vector<hector_perception_msgs::PerceptionData> polygonGroup;
  if(contours.size() != 0) {
    for(int k=0; k < detectionLimit_; k++) {
      for(int j=0; j < areas.size(); j++) {
        if( areas[j] > largest_area ) {
          largest_area = areas[j];
          largest_contour_index = j;               //Store the index of largest contour
          bounding_rect = cv::boundingRect( contours[j] ); // Find the bounding rectangle for biggest contour
        }
      }
      if(areas[largest_contour_index] >= min_area_ && areas[largest_contour_index] <= max_area_) {
        cv::rectangle( frame, bounding_rect.tl(), bounding_rect.br(), cv::Scalar( 0, 0, 255 ), 2, 8, 0 );

        //Create a Polygon consisting of Point32 points for publishing
        geometry_msgs::Point32 p1, p2, p3, p4;
        p1.x = bounding_rect.tl().x;
        p1.y = bounding_rect.tl().y;
        p4.x = bounding_rect.br().x;
        p4.y = bounding_rect.br().y;
        p2.x = p4.x;
        p2.y = p1.y;
        p3.x = p1.x;
        p3.y = p4.y;

        std::vector<geometry_msgs::Point32> recPoints;
        recPoints.push_back(p1);
        recPoints.push_back(p2);
        recPoints.push_back(p4);
        recPoints.push_back(p3);

        geometry_msgs::Polygon polygon;
        polygon.points = recPoints;
        hector_perception_msgs::PerceptionData perceptionData;
        perceptionData.percept_name = "motion" + boost::lexical_cast<std::string>(k);
        perceptionData.polygon = polygon;

        polygonGroup.push_back(perceptionData);
      }
      areas[largest_contour_index] = -1;
      largest_area = 0;
    }
    if (image_perception_pub.getNumSubscribers() > 0) {
      hector_perception_msgs::PerceptionDataArray polygonPerceptionArray;
      polygonPerceptionArray.header.stamp = ros::Time::now();
      polygonPerceptionArray.perceptionType = "motion";
      polygonPerceptionArray.perceptionList = polygonGroup;
      image_perception_pub.publish(polygonPerceptionArray);
    }
  }
  sensor_msgs::CameraInfo::Ptr info;
  info.reset(new sensor_msgs::CameraInfo());
  info->header = img->header;

  if(image_background_subtracted_pub_.getNumSubscribers() > 0) {
    cv_bridge::CvImage cvImg;
    cvImg.image = fgimg_orig;
    cvImg.header = img->header;
    cvImg.encoding = sensor_msgs::image_encodings::MONO8;
    image_background_subtracted_pub_.publish(cvImg.toImageMsg(), info);
  }

  if(image_motion_pub_.getNumSubscribers() > 0) {
    cv_bridge::CvImage cvImg;
    cvImg.image = thresholded;
    cvImg.header = img->header;
    cvImg.encoding = sensor_msgs::image_encodings::MONO8;
    image_motion_pub_.publish(cvImg.toImageMsg(), info);
  }

  if(image_detected_pub_.getNumSubscribers() > 0) {
    cv_bridge::CvImage cvImg;
    cvImg.image = frame;
    cvImg.header = img->header;
    cvImg.encoding = sensor_msgs::image_encodings::BGR8;
    image_detected_pub_.publish(cvImg.toImageMsg(), info);
  }
}

void MotionDetection::publishEnableStatus() {
  std_msgs::Bool bool_msg;
  bool_msg.data = enabled_;
  enabled_pub_.publish(bool_msg);

  std::string enabled_string;
  if (enabled_) {
    enabled_string = "Enabled";
  } else {
    enabled_string = "Disabled";
  }
  ROS_INFO_STREAM(enabled_string << " hector_motion_detection.");
}

void MotionDetection::dynRecParamCallback(MotionDetectionConfig &config, uint32_t level)
{
  bg_subtractor_->setNMixtures(3);
  shadows_ = config.motion_detect_shadows;
  bg_subtractor_->setDetectShadows(shadows_);
  min_area_ = config.motion_detect_min_area;
  max_area_ = config.motion_detect_max_area;
  detectionLimit_ = config.motion_detect_detectionLimit;
  debug_contours_ = config.motion_detect_debug_contours;
  erosion_iterations_ = config.motion_detect_erosion;
  dilation_iterations_ = config.motion_detect_dilation;
  learning_rate_ = config.learning_rate;
  automatic_learning_rate_ = config.automatic_learning_rate;
  moving_average_weight_ = config.moving_average_weight;
  activation_threshold_ = config.activation_threshold;
}

}
