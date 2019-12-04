#include "heat_detection.h"

HeatDetection::HeatDetection(ros::NodeHandle& n,ros::NodeHandle& p_n)
: image_sub_(p_n, "thermal/image", 1),
  cam_info_sub_(p_n, "thermal/camera_info", 1),
  image_mapped_sub_(p_n, "thermal/image_mapped", 1),
  time_sync_(SyncPolicy(10), image_sub_, cam_info_sub_, image_mapped_sub_)
{

    processing_enabled_ = true;

    processing_enabled_pub_= p_n.advertise<std_msgs::Bool>("processing_enabled", 2, true);
    processing_enabled_sub_ = p_n.subscribe("set_processing_enabled", 1, &HeatDetection::setProcessingEnabledCb, this);

    image_count_ = 0;

    //ros::NodeHandle n;
    //ros::NodeHandle p_n("~");//private nh
    image_transport::ImageTransport it(n);
    image_transport::ImageTransport p_it(p_n);

    min_temp_img_ =  10.0;
    max_temp_img_ = 200.0;
    mappingDefined_ = true;

    blob_temperature_ = -1.0;
    perform_measurement_ = false;

    //sub_ = it.subscribeCamera("thermal/image", 1, &HeatDetection::imageCallback,this);

    time_sync_.registerCallback( boost::bind(&HeatDetection::imageCallback, this, _1, _2, _3));

    //sub_mapping_ = n.subscribe("thermal/mapping",1, &HeatDetection::mappingCallback,this);

    dyn_rec_server_.reset(new ReconfigureServer(config_mutex_, p_n));
    dyn_rec_server_->setCallback(boost::bind(&HeatDetection::dynRecParamCallback, this, _1, _2));


    pub_ = n.advertise<hector_worldmodel_msgs::ImagePercept>("image_percept",20);
    pub_detection_ = p_it.advertiseCamera("image", 10);

    //get_measurement_server_= p_n.advertiseService("get_heat_measurement", &HeatDetection::getMeasurementSrvCallback, this);

    this->publishProcessingEnabledState();
}

HeatDetection::~HeatDetection(){}

void HeatDetection::setProcessingEnabledCb(const std_msgs::Bool& msg)
{
  processing_enabled_ = msg.data;
  this->publishProcessingEnabledState();
}

void HeatDetection::publishProcessingEnabledState()
{
  std_msgs::Bool msg;
  msg.data = processing_enabled_;
  processing_enabled_pub_.publish(msg);
}

/*
bool HeatDetection::getMeasurementSrvCallback(argo_vision_msgs::GetMeasurement::Request &req,
                    argo_vision_msgs::GetMeasurement::Response &res){
    blob_temperature_ = -1.0;
    perform_measurement_ = true;

    res.value = blob_temperature_;

    return true;
}
*/

void HeatDetection::imageCallback(const sensor_msgs::ImageConstPtr& img, const sensor_msgs::CameraInfoConstPtr& info,
  const sensor_msgs::ImageConstPtr& img_mapped)
{
//if(debug_){
//    ROS_INFO("count: %i", ++image_count_);
//}
    /*if(perform_measurement_) {
        perform_measurement_ = false;
    } else {
        return;
    }*/
    //std::cout << "Processing image..." << std::endl;

    if (!processing_enabled_)
    {
      return;
    }


    if (!mappingDefined_){
        ROS_WARN("Error: Mapping undefined -> cannot perform detection");
    }else{

     //Read image with cvbridge
     cv_bridge::CvImageConstPtr cv_ptr;
     cv_ptr = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::MONO8);
     cv::Mat img_filtered(cv_ptr->image);

     //Read image with cvbridge
     cv_bridge::CvImagePtr cv_mapped_ptr;
     cv_mapped_ptr = cv_bridge::toCvCopy(img_mapped, sensor_msgs::image_encodings::RGB8);
     cv::Mat cv_img_mapped(cv_mapped_ptr->image);

     if ((img_thres_.rows != static_cast<int>(img->height)) || (img_thres_.cols != static_cast<int>(img->width))){
       img_thres_min_ = cv::Mat (img->height, img->width,CV_8UC1,1);
       img_thres_max_ = cv::Mat (img->height, img->width,CV_8UC1,1);
       img_thres_ = cv::Mat (img->height, img->width,CV_8UC1,1);
    }


   //Perform thresholding

   //Define image thresholds for victim detection
   int minThreshold = (int)std::max(std::min(((minTempVictim_ - min_temp_img_) *(256.0/( max_temp_img_ -  min_temp_img_))),255.0),0.0);
   int maxThreshold = (int)std::max(std::min(((maxTempVictim_ -min_temp_img_) *(256.0/( max_temp_img_ -  min_temp_img_))),255.0),0.0);

   cv::threshold(img_filtered,img_thres_min_,minThreshold,1,cv::THRESH_BINARY);
   cv::threshold(img_filtered,img_thres_max_,maxThreshold,1,cv::THRESH_BINARY_INV);

   //Element-wise multiplication to obtain an image with respect to both thresholds
   IplImage img_thres_min_ipl = img_thres_min_;
   IplImage img_thres_max_ipl = img_thres_max_;
   IplImage img_thres_ipl = img_thres_;

   cvMul(&img_thres_min_ipl, &img_thres_max_ipl, &img_thres_ipl, 255);
   //Perform blob detection
   cv::SimpleBlobDetector::Params params;
   params.filterByColor = true;
   params.blobColor = 255;
   params.minDistBetweenBlobs = minDistBetweenBlobs_;
   params.filterByArea = true;
   params.minArea = minAreaVictim_;
   params.maxArea = img_filtered.rows * img_filtered.cols;
   params.filterByCircularity = false;
   params.filterByColor = false;
   params.filterByConvexity = false;
   params.filterByInertia = false;

   cv::Ptr<cv::SimpleBlobDetector> blob_detector = cv::SimpleBlobDetector::create(params);

   //cv::SimpleBlobDetector blob_detector(params);
   std::vector<cv::KeyPoint> keypoints;
   keypoints.clear();
   blob_detector->detect(img_thres_,keypoints);
   //Publish results
   hector_worldmodel_msgs::ImagePercept ip;

   ip.header= img->header;
   ip.info.class_id = perceptClassId_;
   ip.info.class_support = 1;
   ip.camera_info =  *info;

   blob_temperature_ = -1.0;
   for(unsigned int i=0; i<keypoints.size();i++)
   {
       cv::KeyPoint k = keypoints.at(i);
       ip.x = k.pt.x;
       ip.y = k.pt.y;
       ROS_DEBUG("Heat blob found at image coord: (%f, %f)", ip.x, ip.y);

       cv::Rect rect_image(0, 0, img_filtered.cols, img_filtered.rows);
       cv::Rect rect_roi(k.pt.x - (k.size - 1)/2, k.pt.y - (k.size - 1)/2, k.size - 1, k.size - 1);

       //See http://stackoverflow.com/questions/29120231/how-to-verify-if-rect-is-inside-cvmat-in-opencv
       bool is_inside = (rect_roi & rect_image) == rect_roi;

       if (!is_inside){
         ROS_ERROR("ROI image would be partly outside image border, aborting further processing!");
         continue;
       }

       const cv::Mat roi = img_filtered(rect_roi);
       int histSize = 256;
       float range[] = { 0, 256 };
       const float* histRange = { range };
       cv::Mat hist;
       cv::calcHist(&roi, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

       float max_value = 0;
       for(int j=0; j<histSize; j++) {
           float h = hist.at<float>(j);
           //std::cout << "value: " << h << std::endl;
           if (max_value <= h) {
               max_value = h;
               if (j > blob_temperature_) {
                   blob_temperature_ = j;
               }
           }
       }
       
       //ip.info.data.push_back(blob_temperature_);
       pub_.publish(ip);
       //std::cout << hist << std::endl;
       //std::cout << "Result: " << max_value << " --> " << blob_temperature_ << std::endl;
   }
   if (keypoints.size() == 0) {
       //std::cout << "No blob detected" << std::endl;
   }


   if(pub_detection_.getNumSubscribers() > 0){

      //Display Keypoints
       for(unsigned int i = 0; i < keypoints.size(); i++){
           if (keypoints.at(i).size > 1){

             int half_size = keypoints.at(i).size/2;
             cv::rectangle(cv_img_mapped, cv::Point(keypoints.at(i).pt.x - half_size, keypoints.at(i).pt.y - half_size),
               cv::Point(keypoints.at(i).pt.x + half_size, keypoints.at(i).pt.y + half_size), cv::Scalar(255), 3);

           }
       }

       cv_bridge::CvImage cvImg;
       cvImg.image = cv_img_mapped;
       
       cvImg.header = img->header;
       cvImg.encoding = sensor_msgs::image_encodings::RGB8;
       pub_detection_.publish(cvImg.toImageMsg(),info);
    }
}
}

/*
void HeatDetection::mappingCallback(const thermaleye_msgs::Mapping& mapping){
   mapping_ = mapping;
   mappingDefined_ = true;
   ROS_INFO("Mapping received");
}
*/

void HeatDetection::dynRecParamCallback(HeatDetectionConfig &config, uint32_t level)
{
  minTempVictim_ = config.min_temp_detection;
  maxTempVictim_ = config.max_temp_detection;
  minAreaVictim_ = config.min_area_detection;
  minDistBetweenBlobs_ = config.min_dist_between_blobs;
  perceptClassId_ = config.percept_class_id;
}

