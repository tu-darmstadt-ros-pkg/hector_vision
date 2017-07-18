#include <hector_detection_aggregator/detection_aggregator.h>

namespace hector_detection_aggregator
{
DetectionAggregator::DetectionAggregator()
{
    storage_duration_ = ros::Duration(10);

    ros::NodeHandle n;
    ros::NodeHandle p_n("~"); //private nh

    img_current_grey_ptr_.reset();
    img_current_col_ptr_.reset();

    image_transport::ImageTransport it(n);
    image_transport::ImageTransport image_detected_it(p_n);

    image_sub_ = it.subscribe("/arm_rgbd_cam/rgb/image_raw", 1 , &DetectionAggregator::imageCallback, this);
    image_percept_sub_ = n.subscribe("/detection/visual_detection", 1 , &DetectionAggregator::imageDetectionCallback, this);
     dyn_rec_server_.setCallback(boost::bind(&DetectionAggregator::dynRecParamCallback, this, _1, _2));

    image_detected_pub_ = image_detected_it.advertiseCamera("/detection/aggregated_detections_image", 10);

    color_map_["motion"] = cv::Scalar(0,0,255);
    color_map_["qr"] = cv::Scalar(255,0,0);
    color_map_["heat"] = cv::Scalar(0,255,0);
    color_map_["hazmat"] = cv::Scalar(255,255,0);
}

DetectionAggregator::~DetectionAggregator() {}


void DetectionAggregator::updateDetections()
{
    ros::Time storage_threshold;
    if((ros::Time::now().toSec() > storage_duration_.toSec())){
      storage_threshold =ros::Time::now() - storage_duration_;
    }
    std::vector<std::string> keys;
    for(std::map<std::string, hector_perception_msgs::PerceptionDataArray>::iterator it = detection_map_.begin(); it != detection_map_.end(); ++it) {
        keys.push_back(it->first);
    }
    for(std::string& key : keys)
    {
        const hector_perception_msgs::PerceptionDataArray& detection_array = detection_map_[key];
        if(detection_array.header.stamp < storage_threshold)
        {
            auto it = detection_map_.find (key);
            detection_map_.erase (it);
        }
    }
}

void DetectionAggregator::createImage()
{
    if(image_detected_pub_.getNumSubscribers() == 0)
        return;

    updateDetections();

    if(img_current_col_ptr_ )
    {
        cv::Mat img_detected;
        img_current_col_ptr_->image.copyTo(img_detected);

        for(auto& percept_pair : detection_map_)
        {
            hector_perception_msgs::PerceptionDataArray& percept_list = percept_pair.second;
            for(hector_perception_msgs::PerceptionData& percept : percept_list.perceptionList)
            {
                std::vector<cv::Point> polygon;


                cv::Point cv_center_point;
                for(geometry_msgs::Point32& ros_point: percept.polygon.points)
                {
                    cv::Point cv_point;
                    cv_point.x = ros_point.x;
                    cv_point.y = ros_point.y;
                    polygon.push_back(cv_point);
                    cv_center_point += cv_point;
                }
                cv_center_point = cv_center_point*(1./percept.polygon.points.size());
                const cv::Point *pts = (const cv::Point*) cv::Mat(polygon).data;
                int npts = cv::Mat(polygon).rows;

                //ROS_INFO("Type: %s color %f %f %f ",percept_pair.first.c_str(),color_map_[percept_pair.first][0],color_map_[percept_pair.first][1],color_map_[percept_pair.first][2]);
                polylines(img_detected, &pts,&npts, 1,
                          true, 			// draw closed contour (i.e. joint end to start)
                          color_map_[percept_pair.first.c_str()],// colour RGB ordering
                        2, 		        // line thickness
                        CV_AA, 0);
                cv::putText(img_detected,percept.percept_name,cv_center_point,CV_FONT_HERSHEY_PLAIN,2,color_map_[percept_pair.first.c_str()]);
            }
        }

        cv_bridge::CvImage cvImg;
        img_detected.copyTo(cvImg.image);
        //cvImg.header = img->header;
        cvImg.encoding = sensor_msgs::image_encodings::BGR8;
        sensor_msgs::CameraInfo::Ptr info;
        info.reset(new sensor_msgs::CameraInfo());
        // info->header = img->header; todo(kdaun) add info header
        info->header.stamp = ros::Time::now();

        image_detected_pub_.publish(cvImg.toImageMsg(), info);
    }
}

void DetectionAggregator::imageDetectionCallback(const hector_perception_msgs::PerceptionDataArrayConstPtr& percept)
{
    if(image_detected_pub_.getNumSubscribers() == 0)
        return;

    detection_map_[percept->perceptionType] = (*percept);
    //ROS_INFO("Image Percept time %f",(float)(*percept).header.stamp.toSec());
}

void DetectionAggregator::imageCallback(const sensor_msgs::ImageConstPtr& img) //, const sensor_msgs::CameraInfoConstPtr& info)
{
    if(image_detected_pub_.getNumSubscribers() == 0)
        return;

    img_current_grey_ptr_ = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::MONO8);
    img_current_col_ptr_ = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::BGR8);

}



void DetectionAggregator::dynRecParamCallback(HectorDetectionAggregatorConfig &config, uint32_t level)
{
    storage_duration_ = ros::Duration(config.storage_duration);
}


}
