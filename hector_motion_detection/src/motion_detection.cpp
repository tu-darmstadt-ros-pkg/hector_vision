//Author: Matej Zecevic
#include "motion_detection.h"

MotionDetection::MotionDetection()
  : first_image_received_(false)
{
    bg = cv::createBackgroundSubtractorMOG2();

    ros::NodeHandle n;
    ros::NodeHandle p_n("~"); //private nh

    p_n.param("enabled", enabled_, true);

    image_transport::ImageTransport it(n);

    image_sub_ = it.subscribe("image", 1 , &MotionDetection::imageCallback, this);
    enabled_sub_ = p_n.subscribe("enabled", 10, &MotionDetection::enabledCallback, this);

    boost::bind(&MotionDetection::dynRecParamCallback, this, _1, _2);
    dyn_rec_type_ = boost::bind(&MotionDetection::dynRecParamCallback, this, _1, _2);
    dyn_rec_server_.setCallback(dyn_rec_type_);

    image_percept_pub_ = n.advertise<hector_worldmodel_msgs::ImagePercept>("image_percept", 20);
    image_motion_pub_ = it.advertiseCamera("image_motion", 10);
    image_detected_pub_ = it.advertiseCamera("image_detected", 10);

    image_perception_pub = n.advertise<hector_perception_msgs::PerceptionDataArray>("detection/image_detection", 10);
    ROS_INFO("Starting with Motion Detection with MOG2");
    ROS_INFO("debug_contours: %i", debug_contours);
    ROS_INFO("shadows: %i", shadows);
    ROS_INFO("max area: %d", max_area);
    ROS_INFO("min area: %d", min_area);
    ROS_INFO("detection limit: %d", detectionLimit);
    image_transport::ImageTransport image_bg_it(p_n);
    image_background_subtracted_pub_ = image_bg_it.advertiseCamera("image_background_subtracted", 10);
}

MotionDetection::~MotionDetection() {}

void MotionDetection::enabledCallback(const std_msgs::BoolConstPtr& enabled) {
  enabled_ = enabled->data;
}

void MotionDetection::imageCallback(const sensor_msgs::ImageConstPtr& img)
{
  if (!enabled_) {
    return;
  }
  cv_bridge::CvImageConstPtr cv_ptr;
  cv_ptr = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::BGR8);
  cv::Mat img_filtered(cv_ptr->image);

  std::vector<std::vector<cv::Point> > contours;
  cv::Mat frame, fgimg, fgimg_orig;

  img_filtered.copyTo(frame);

  if (automatic_learning_rate_) {
    bg->apply(img_filtered, fgimg);
  } else {
    bg->apply(img_filtered, fgimg, learning_rate_);
  }

  cv::morphologyEx(fgimg, fgimg, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));

  fgimg.copyTo(fgimg_orig);   //for debugging/tuning purposes

  //controlable iterations for morphological operations
  for(int i=0; i < erosion_iterations; i++) {
    cv::erode(fgimg, fgimg, cv::Mat());
  }
  for(int i=0; i < dilation_iterations; i++) {
    cv::dilate(fgimg, fgimg, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));
    //alternatively (tuning): other kernels: e.g. cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10,10))
  }

  if (!first_image_received_) {
    fgimg.copyTo(accumulated_image_);
    first_image_received_ = true;
  } else {
    cv::addWeighted(accumulated_image_, 0.9, fgimg, 0.1, 0.0, accumulated_image_);
  }
  cv::Mat thres;
  cv::threshold(accumulated_image_, thres, 100, 255, cv::THRESH_BINARY);
  cv::findContours (thres, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  if (debug_contours) {
    cv::drawContours(frame, contours, -1, cv::Scalar (0, 0, 255), 2);
  }

  //========Detection of g largest contours=====
  int largest_area=0;
  int largest_contour_index=0;
  cv::Rect bounding_rect;
  std::vector<double> areas( contours.size() );

  for( int i = 0; i < contours.size(); i++ )
  {
    double area = cv::contourArea( contours[i] );  //  Find the area of contour
    areas[i] = area;
  }

  std::vector<hector_perception_msgs::PerceptionData> polygonGroup;
  if(contours.size() != 0) {
    for(int k=0; k < detectionLimit; k++) {
      for(int j=0; j < areas.size(); j++) {
        if( areas[j] > largest_area ) {
          largest_area = areas[j];
          largest_contour_index = j;               //Store the index of largest contour
          bounding_rect = cv::boundingRect( contours[j] ); // Find the bounding rectangle for biggest contour
        }
      }
      if(areas[largest_contour_index] >= min_area && areas[largest_contour_index] <= max_area) {
        cv::rectangle( frame, bounding_rect.tl(), bounding_rect.br(), cv::Scalar( 0, 0, 255 ), 2, 8, 0 );

        //Create a Polygon consisting of Point32 points for publishing
        //std::cout << "BR first: " << bounding_rect.tl() << "BR lasst: " << bounding_rect.br() << std::endl;
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
        //std::cout << "recPoints: " << recPoints[0] << recPoints[1] << recPoints[2] << recPoints[3] << std::endl;

        geometry_msgs::Polygon polygon;
        polygon.points = recPoints;
        //std::cout << "Polygon: " << polygon.points[0] << polygon.points[1] << polygon.points[2] << polygon.points[3] << std::endl;
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
    fgimg_orig.copyTo(cvImg.image);
    cvImg.header = img->header;
    cvImg.encoding = sensor_msgs::image_encodings::MONO8;
    image_background_subtracted_pub_.publish(cvImg.toImageMsg(), info);
  }

  if(image_motion_pub_.getNumSubscribers() > 0) {
    cv_bridge::CvImage cvImg;
    thres.copyTo(cvImg.image);
    cvImg.header = img->header;
    cvImg.encoding = sensor_msgs::image_encodings::MONO8;
    image_motion_pub_.publish(cvImg.toImageMsg(), info);
  }

  if(image_detected_pub_.getNumSubscribers() > 0) {
    cv_bridge::CvImage cvImg;
    frame.copyTo(cvImg.image);
    cvImg.header = img->header;
    cvImg.encoding = sensor_msgs::image_encodings::BGR8;
    image_detected_pub_.publish(cvImg.toImageMsg(), info);
  }
}

void MotionDetection::dynRecParamCallback(MotionDetectionConfig &config, uint32_t level)
{
  bg->setNMixtures(3);
  shadows = config.motion_detect_shadows;
  bg->setDetectShadows(shadows);
  min_area = config.motion_detect_min_area;
  max_area = config.motion_detect_max_area;
  detectionLimit = config.motion_detect_detectionLimit;
  debug_contours = config.motion_detect_debug_contours;
  erosion_iterations = config.motion_detect_erosion;
  dilation_iterations = config.motion_detect_dilation;
  learning_rate_ = config.learning_rate;
  automatic_learning_rate_ = config.automatic_learning_rate;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "motion_detection");
  MotionDetection md;
  ros::spin();
}

