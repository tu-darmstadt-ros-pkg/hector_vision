#include <hector_detection_aggregator/detection_aggregator.h>

namespace hector_detection_aggregator
{
DetectionAggregator::DetectionAggregator()
{
    ros::NodeHandle n;
    ros::NodeHandle p_n("~"); //private nh

    double storage_duration;
    n.param("storage_duration", storage_duration, 1.0);
    storage_duration_ = ros::Duration(storage_duration);

    img_current_ptr_.reset();
    img_current_col_ptr_.reset();

    image_transport::ImageTransport it(n);
    image_transport::ImageTransport image_detected_it(p_n);

    image_sub_ = it.subscribe("arm_rgbd_cam/rgb/image_raw", 1 , &DetectionAggregator::imageCallback, this);
    image_percept_sub_ = n.subscribe("perception/image_percept", 1 , &DetectionAggregator::imagePerceptCallback, this);
     dyn_rec_server_.setCallback(boost::bind(&DetectionAggregator::dynRecParamCallback, this, _1, _2));

    image_detected_pub_ = image_detected_it.advertiseCamera("image_aggregated_percepts", 10);

    color_map_["motion"] = cv::Scalar(0,0,255);
    color_map_["qr"] = cv::Scalar(255,0,0);
    color_map_["heat"] = cv::Scalar(0,255,0);
    color_map_["hazmat"] = cv::Scalar(255,255,0);
}

DetectionAggregator::~DetectionAggregator() {}


void DetectionAggregator::updatePercepts()
{

    ros::Time storage_threshold = ros::Time::now() - storage_duration_;
    std::vector<std::string> keys;
    for(std::map<std::string, hector_perception_msgs::PerceptionDataArray>::iterator it = percept_storage_.begin(); it != percept_storage_.end(); ++it) {
        keys.push_back(it->first);
        std::cout << it->first << "\n";
    }
    for(std::string& key : keys)
    {
        const hector_perception_msgs::PerceptionDataArray& percept_array = percept_storage_[key];
        if(percept_array.header.stamp < storage_threshold)
        {
            //ROS_INFO("outdated %f",percept_array.header.stamp.toSec());
            auto it = percept_storage_.find (key);             // by iterator (b), leaves acdefghi.
            percept_storage_.erase (it);
        }
    }
}

void DetectionAggregator::createImage()
{
    updatePercepts();

    if(img_current_col_ptr_ )
    {
        cv::Mat img_detected;
        img_current_col_ptr_->image.copyTo(img_detected);

        for(auto& percept_pair : percept_storage_)
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
                          color_map_[percept_pair.first.c_str()],// colour RGB ordering (here = green)
                        4, 		        // line thickness
                        CV_AA, 0);
                cv::putText(img_detected,percept.percept_name,cv_center_point,CV_FONT_HERSHEY_PLAIN,3,color_map_[percept_pair.first.c_str()]);
            }
        }

        cv_bridge::CvImage cvImg;
        img_detected.copyTo(cvImg.image);
        //cvImg.header = img->header;
        cvImg.encoding = sensor_msgs::image_encodings::BGR8;
        sensor_msgs::CameraInfo::Ptr info;
        info.reset(new sensor_msgs::CameraInfo());
        // info->header = img->header; todo add info header

        image_detected_pub_.publish(cvImg.toImageMsg(), info);
    }
    // percept_storage_.clear();
}

void DetectionAggregator::imagePerceptCallback(const hector_perception_msgs::PerceptionDataArrayConstPtr& percept)
{
    percept_storage_[percept->perceptionType]=(*percept);
    //ROS_INFO("Image Percept time %f",(float)(*percept).header.stamp.toSec());
}

void DetectionAggregator::imageCallback(const sensor_msgs::ImageConstPtr& img) //, const sensor_msgs::CameraInfoConstPtr& info)
{
    // get image
    img_current_ptr_ = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::MONO8);
    img_current_col_ptr_ = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::BGR8);



    /*
        if (number_of_changes && max_percept_size > percept_size && percept_size > min_percept_size && density > min_density)
        {
            cv::rectangle(img_detected, cv::Rect(cv::Point(min_x, min_y), cv::Point(max_x, max_y)), CV_RGB(255,0,0), 5);
        }

        //cv::imshow("view", img_current_ptr_->image);
        sensor_msgs::CameraInfo::Ptr info;
        info.reset(new sensor_msgs::CameraInfo());
        info->header = img->header;

        if(image_motion_pub_.getNumSubscribers() > 0)
        {
            cv_bridge::CvImage cvImg;
            img_motion.copyTo(cvImg.image);
            cvImg.header = img->header;
            cvImg.encoding = sensor_msgs::image_encodings::MONO8;
            image_motion_pub_.publish(cvImg.toImageMsg(), info);
        }

        if(image_detected_pub_.getNumSubscribers() > 0)
        {
            cv_bridge::CvImage cvImg;
            img_detected.copyTo(cvImg.image);
            cvImg.header = img->header;
            cvImg.encoding = sensor_msgs::image_encodings::BGR8;
            image_detected_pub_.publish(cvImg.toImageMsg(), info);
        }
    }
    */



}



void DetectionAggregator::dynRecParamCallback(HectorDetectionAggregatorConfig &config, uint32_t level)
{
    storage_duration_ = ros::Duration(config.storage_duration);

}


}
