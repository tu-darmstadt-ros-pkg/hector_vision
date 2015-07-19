#include "motion_detection.h"

MotionDetection::MotionDetection()
{
  ros::NodeHandle nh;
  ros::NodeHandle p_nh("~"); //private nh

  image_transport::ImageTransport it(nh);
  image_transport::ImageTransport image_motion_it(p_nh);
  image_transport::ImageTransport image_detected_it(p_nh);

  // subscribe topics
  image_sub_ = it.subscribe("opstation/rgb/image_color", 1, &MotionDetection::imageCallback, this);
  //image_sub_ = it.subscribe("/openni/rgb/image_color", 1, &MotionDetection::imageCallback, this);

  // advertise topics
  image_percept_pub_ = nh.advertise<hector_worldmodel_msgs::ImagePercept>("image_percept", 20);
  image_motion_pub_ = image_motion_it.advertiseCamera("image_motion", 10);
  image_detected_pub_ = image_detected_it.advertiseCamera("image_detected", 10);

  // dynamic reconfigure
  dyn_rec_server_.setCallback(boost::bind(&MotionDetection::dynRecParamCallback, this, _1, _2));

  update_timer = nh.createTimer(ros::Duration(0.04), &MotionDetection::update, this);
}

MotionDetection::~MotionDetection()
{
}

void MotionDetection::colorizeDepth(const cv::Mat& gray, cv::Mat& rgb) const
{
  double maxDisp = 255;
  float S= 1.0f;
  float V= 1.0f ;

  rgb.create(gray.size(), CV_8UC3);
  rgb = cv::Scalar::all(0);

  if (maxDisp < 1)
    return;

  for (int y = 0; y < gray.rows; y++)
  {
    for (int x = 0; x < gray.cols; x++)
    {
      uchar d = gray.at<uchar>(y,x);
      unsigned int H = 255 - ((uchar)maxDisp - d) * 280/ (uchar)maxDisp;
      unsigned int hi = (H/60) % 6;

      float f = H/60.f - H/60;
      float p = V * (1 - S);
      float q = V * (1 - f * S);
      float t = V * (1 - (1 - f) * S);

      cv::Point3f res;

      if (hi == 0) //R = V,  G = t,  B = p
        res = cv::Point3f( p, t, V );
      if (hi == 1) // R = q, G = V,  B = p
        res = cv::Point3f( p, V, q );
      if (hi == 2) // R = p, G = V,  B = t
        res = cv::Point3f( t, V, p );
      if (hi == 3) // R = p, G = q,  B = V
        res = cv::Point3f( V, q, p );
      if (hi == 4) // R = t, G = p,  B = V
        res = cv::Point3f( V, p, t );
      if (hi == 5) // R = V, G = p,  B = q
        res = cv::Point3f( q, p, V );

      uchar b = (uchar)(std::max(0.f, std::min (res.x, 1.f)) * 255.f);
      uchar g = (uchar)(std::max(0.f, std::min (res.y, 1.f)) * 255.f);
      uchar r = (uchar)(std::max(0.f, std::min (res.z, 1.f)) * 255.f);

      rgb.at<cv::Point3_<uchar> >(y,x) = cv::Point3_<uchar>(b, g, r);
    }
  }
}

void MotionDetection::drawOpticalFlowVectors(cv::Mat& img, const cv::Mat& optical_flow, int step, const cv::Scalar& color) const
{
  for (int y = 0; y < optical_flow.rows; y += step)
  {
    for (int x = 0; x < optical_flow.cols; x += step)
    {
      const cv::Point2f& fxy = optical_flow.at<cv::Point2f>(y, x);
      line(img, cv::Point(x,y), cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), color);
      circle(img, cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), 1, color, -1);
    }
  }
}

void MotionDetection::computeOpticalFlow(const cv::Mat& prev_img, const cv::Mat& cur_img, cv::Mat& optical_flow, bool use_initial_flow, bool filter) const
{
  // compute optical flow
  int flags = 0;
  if (use_initial_flow && optical_flow.rows == prev_img.rows && optical_flow.cols == prev_img.cols && optical_flow.type() == CV_32FC2)
    flags = cv::OPTFLOW_USE_INITIAL_FLOW;
  else
    optical_flow = cv::Mat(prev_img.rows, prev_img.cols, CV_32FC2);

  calcOpticalFlowFarneback(prev_img, cur_img, optical_flow, 0.5, 2, 15, 3, 5, 1.2, flags);

  if (filter)
  {
    // zero-mean data
    cv::Point2f mean;
    mean.x = 0;
    mean.y = 0;
    for(int y = 0; y < optical_flow.rows; y++)
    {
      for(int x = 0; x < optical_flow.cols; x++)
        mean += optical_flow.at<cv::Point2f>(y, x);
    }

    mean.x /= static_cast<float>(optical_flow.rows*optical_flow.cols);
    mean.y /= static_cast<float>(optical_flow.rows*optical_flow.cols);
    float mean_length = std::sqrt(mean.x*mean.x + mean.y*mean.y);

    float max_cos = cos(M_PI_2);
    for(int y = 0; y < optical_flow.rows; y++)
    {
      for(int x = 0; x < optical_flow.cols; x++)
      {
        cv::Point2f& p = optical_flow.at<cv::Point2f>(y, x);

        float cos_pm = (p.x*mean.x+p.y*mean.y) / (std::sqrt(p.x*p.x + p.y*p.y)*mean_length);
        if (cos_pm > max_cos)
          p -= mean;
      }
    }
  }
}

void MotionDetection::computeOpticalFlowMagnitude(const cv::Mat& optical_flow, cv::Mat& optical_flow_mag) const
{
  optical_flow_mag = cv::Mat(optical_flow.size(), CV_8UC1);

  for (int y = 0; y < optical_flow.rows; y++)
  {
    for (int x = 0; x < optical_flow.cols; x++)
    {
      const cv::Point2f& fxy = optical_flow.at<cv::Point2f>(y, x);
      double magnitude = std::min(std::sqrt(fxy.x*fxy.x+fxy.y*fxy.y)/motion_detect_inv_sensivity_, 1.0);
      optical_flow_mag.at<uchar>(y, x) = static_cast<uchar>(magnitude*255);
    }
  }
}

void MotionDetection::drawBlobs(cv::Mat& img, const KeyPoints& keypoints, double scale) const
{
  if (img.rows == 0 || img.cols == 0)
    return;

  double line_width = 2.0*scale;
  for (std::vector<cv::KeyPoint>::const_iterator itr = keypoints.begin(); itr != keypoints.end(); itr++)
  {
    const cv::KeyPoint& keypoint = *itr;

    if (keypoint.size > 1)
    {
      float width = keypoint.size;
      float height = keypoint.size;

      const cv::Point2f& p = keypoint.pt;
      cv::rectangle(img, cv::Rect(cv::Point(static_cast<int>((p.x-0.5*width-line_width)*scale), static_cast<int>((p.y-0.5*height-line_width)*scale)), cv::Point(static_cast<int>((p.x+0.5*width+line_width)*scale), static_cast<int>((p.y+0.5*height+line_width)*scale))), CV_RGB(255,0,0), line_width);
    }
  }
}

void MotionDetection::detectBlobs(const cv::Mat& img, KeyPoints& keypoints) const
{
  // Perform thresholding
  cv::Mat img_thresh;
  cv::threshold(img, img_thresh, motion_detect_threshold_, 255, CV_THRESH_BINARY);

  // Dilate detected area
  cv::Mat kernel_dil = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(motion_detect_dilation_size_, motion_detect_dilation_size_));
  cv::dilate(img_thresh, img_thresh, kernel_dil);

  // Perform blob detection
  cv::SimpleBlobDetector::Params params;
  params.minDistBetweenBlobs = motion_detect_min_blob_dist_;
  params.filterByArea = true;
  params.minArea = motion_detect_min_area_;
  params.maxArea = img_thresh.rows * img_thresh.cols;
  params.filterByCircularity = false;
  params.filterByColor = true;
  params.blobColor = 255;
  params.filterByConvexity = false;
  params.filterByInertia = false;

  cv::SimpleBlobDetector blob_detector(params);
  keypoints.clear();
  blob_detector.detect(img_thresh, keypoints);

#ifdef DEBUG
  cv::imshow("blob_detector_input", img_thresh);
#endif
}

void MotionDetection::update(const ros::TimerEvent& /*event*/)
{
  if (!last_img)
    return;

  // get image
  cv_bridge::CvImageConstPtr img_next_ptr(cv_bridge::toCvShare(last_img, sensor_msgs::image_encodings::MONO8));
  cv_bridge::CvImageConstPtr img_next_col_ptr(cv_bridge::toCvShare(last_img, sensor_msgs::image_encodings::BGR8));

  last_img.reset();

  if (img_prev_ptr_ && img_prev_col_ptr_)
  {
    cv::Mat img_prev;
    cv::Mat img_next;
    cv::Mat img_next_col;

    cv::Size img_size(img_next_ptr->image.cols / motion_detect_downscale_factor_, img_next_ptr->image.rows / motion_detect_downscale_factor_);
    cv::resize(img_prev_ptr_->image, img_prev, img_size);
    cv::resize(img_next_ptr->image, img_next, img_size);
    cv::resize(img_next_col_ptr->image, img_next_col, img_size);

    computeOpticalFlow(img_prev, img_next, optical_flow, motion_detect_use_initial_flow_, motion_detect_image_flow_filter_);

    cv::Mat optical_flow_mag;
    computeOpticalFlowMagnitude(optical_flow, optical_flow_mag);

#ifdef DEBUG
    cv::Mat optical_flow_img;
    img_next_col.copyTo(optical_flow_img);
    drawOpticalFlowVectors(optical_flow_img, optical_flow);
    cv::resize(optical_flow_img, optical_flow_img, img_prev_ptr_->image.size());
    cv::imshow("flow", optical_flow_img);
#endif

    cv::Mat optical_flow_mag_img;
    colorizeDepth(optical_flow_mag, optical_flow_mag_img);

#ifdef DEBUG
    cv::resize(optical_flow_mag_img, optical_flow_mag_img, img_prev_ptr_->image.size());
    cv::imshow("magnitude", optical_flow_mag_img);
#endif

    cv::Mat total_flow;
    optical_flow_mag.copyTo(total_flow);

    if (flow_history.size() < static_cast<size_t>(motion_detect_flow_history_size_))
    {
      flow_history.push_front(optical_flow_mag);
      return;
    }
    else
    {
      for (std::list<cv::Mat>::const_iterator itr = flow_history.begin(); itr != flow_history.end(); itr++)
      {
        cv::Mat img_thresh;
        cv::threshold(*itr, img_thresh, motion_detect_threshold_, 255, CV_THRESH_BINARY);
        cv::bitwise_or(total_flow, img_thresh, total_flow);
      }

      flow_history.push_front(optical_flow_mag);

      while (flow_history.size() > static_cast<size_t>(motion_detect_flow_history_size_))
        flow_history.pop_back();
    }

    KeyPoints keypoints;
    detectBlobs(total_flow, keypoints);

    // generate image where the detected movement is encircled by a rectangle
    cv::Mat img_detected;
    img_next_col_ptr->image.copyTo(img_detected);
    drawBlobs(img_detected, keypoints, motion_detect_downscale_factor_);

#ifdef DEBUG
    cv::imshow("view", img_detected);
#endif

    sensor_msgs::CameraInfo::Ptr info;
    info.reset(new sensor_msgs::CameraInfo());
    info->header = img_next_ptr->header;

    if (image_motion_pub_.getNumSubscribers() > 0)
    {
      cv_bridge::CvImage cvImg;
      optical_flow_mag_img.copyTo(cvImg.image);
      cvImg.header = img_next_ptr->header;
      cvImg.encoding = sensor_msgs::image_encodings::BGR8;
      image_motion_pub_.publish(cvImg.toImageMsg(), info);
    }

    if (image_detected_pub_.getNumSubscribers() > 0)
    {
      cv_bridge::CvImage cvImg;
      img_detected.copyTo(cvImg.image);
      cvImg.header = img_next_ptr->header;
      cvImg.encoding = sensor_msgs::image_encodings::BGR8;
      image_detected_pub_.publish(cvImg.toImageMsg(), info);
    }
  }

  // shift image buffers
  img_prev_ptr_= img_next_ptr;
  img_prev_col_ptr_ = img_next_col_ptr;
}

void MotionDetection::imageCallback(const sensor_msgs::ImageConstPtr& img)
{
  last_img = img;
}

void MotionDetection::dynRecParamCallback(MotionDetectionConfig& config, uint32_t level)
{
  motion_detect_downscale_factor_ = config.motion_detect_downscale_factor;
  motion_detect_inv_sensivity_ = config.motion_detect_inv_sensivity;
  motion_detect_use_initial_flow_ = config.motion_detect_use_initial_flow;
  motion_detect_image_flow_filter_ = config.motion_detect_image_flow_filter;
  motion_detect_threshold_ = config.motion_detect_threshold;
  motion_detect_min_area_ = config.motion_detect_min_area;
  motion_detect_min_blob_dist_ = config.motion_detect_min_blob_dist;
  motion_detect_dilation_size_ = config.motion_detect_dilation_size;
  motion_detect_flow_history_size_ = config.motion_detect_flow_history_size;
  percept_class_id_ = config.percept_class_id;

  flow_history.clear();
}

int main(int argc, char **argv)
{
#ifdef DEBUG
  cv::namedWindow("view");
  cv::namedWindow("flow");
  cv::namedWindow("magnitude");
  cv::namedWindow("blob_detector_input");
  cv::startWindowThread();
#endif

  ros::init(argc, argv, "motion_detection");
  MotionDetection md;
  ros::spin();

#ifdef DEBUG
  cv::destroyWindow("view");
  cv::destroyWindow("flow");
  cv::destroyWindow("magnitude");
  cv::destroyWindow("blob_detector_input");
#endif

  return 0;
}

