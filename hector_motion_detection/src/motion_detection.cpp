//Author: Matej Zecevic
#include "motion_detection.h"

MotionDetection::MotionDetection()
{
    ros::NodeHandle n;
    ros::NodeHandle p_n("~"); //private nh

    img_prev_ptr_.reset();
    img_current_ptr_.reset();

    image_transport::ImageTransport it(n);
    image_transport::ImageTransport image_motion_it(p_n);
    image_transport::ImageTransport image_detected_it(p_n);

    //camera_sub_ = it.subscribeCamera("opstation/rgb/image_color", 1, &MotionDetection::imageCallback, this);

    image_sub_ = it.subscribe("/arm_rgbd_cam/rgb/image_raw", 1 , &MotionDetection::imageCallback, this);

    boost::bind(&MotionDetection::dynRecParamCallback, this, _1, _2);
    dyn_rec_type_ = boost::bind(&MotionDetection::dynRecParamCallback, this, _1, _2);
    dyn_rec_server_.setCallback(dyn_rec_type_);

    image_percept_pub_ = n.advertise<hector_worldmodel_msgs::ImagePercept>("image_percept", 20);
    image_motion_pub_ = image_motion_it.advertiseCamera("image_motion", 10);
    image_detected_pub_ = image_detected_it.advertiseCamera("image_detected", 10);

    image_perception_pub = n.advertise<hector_perception_msgs::PerceptionDataArray>("perception/image_percept", 10);
    ROS_INFO("Starting with Motion Detection with MOG2");
    ROS_INFO("max area: %d", max_area);
    ROS_INFO("min area: %d", min_area);
    ROS_INFO("detection limit: %d", detectionLimit);
    image_transport::ImageTransport image_bg_it(p_n);
    image_background_subtracted_pub_ = image_bg_it.advertiseCamera("image_background_subtracted", 10);
}

MotionDetection::~MotionDetection() {}

void MotionDetection::imageCallback(const sensor_msgs::ImageConstPtr& img) //, const sensor_msgs::CameraInfoConstPtr& info)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    cv_ptr = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::BGR8);
    cv::Mat img_filtered(cv_ptr->image);

    std::vector < std::vector < cv::Point > >contours;
    cv::Mat frame, fgimg, backgroundImage, fgimg_orig;

    img_filtered.copyTo(fgimg);
    img_filtered.copyTo(frame);

    bg.operator()(img_filtered, fgimg);

    cv::morphologyEx(fgimg, fgimg, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));

    fgimg.copyTo(fgimg_orig);   //for debugging/tuning purposes
    bg.getBackgroundImage (backgroundImage);


    //controlable iterations for morphological operations
    for(int i=0; i < erosion_iterations; i++){
        cv::erode (fgimg, fgimg, cv::Mat());
    }
    for(int i=0; i < dilation_iterations; i++){
        cv::dilate (fgimg, fgimg, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3))); //alternatively (tuning): other kernels: e.g. cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10,10))
    }

    cv::findContours (fgimg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    //cv::imshow("Binary", fgimg);
    //cv::drawContours (frame, contours, -1, cv::Scalar (0, 0, 255), 2);

    //========Detection of g largest contours=====
    int largest_area=0;
    int largest_contour_index=0;
    cv::Rect bounding_rect;
    std::vector<cv::Rect> boundRect( contours.size() );
    std::vector<cv::vector<cv::Point> > contours_poly( contours.size() );
    std::vector<cv::Point2f>center( contours.size() );
    std::vector<float>radius( contours.size() );
    std::vector<double> areas( contours.size() );

    for( int i = 0; i < contours.size(); i++ )
    {
        double area = cv::contourArea( contours[i] );  //  Find the area of contour
        areas[i] = area;
    }

    std::vector<cv::Rect> rectGroup(detectionLimit);

    std::vector<hector_perception_msgs::PerceptionData> polygonGroup;

    if(contours.size() != 0){
        for(int k=0; k < detectionLimit; k++){
            for(int j=0; j < areas.size(); j++){
                if( areas[j] > largest_area )
                {
                    largest_area = areas[j];
                    largest_contour_index = j;               //Store the index of largest contour
                    bounding_rect = cv::boundingRect( contours[j] ); // Find the bounding rectangle for biggest contour
                }
            }

            //std::cout << "rectGroup after with " << k << " : " << rectGroup[0] << std::endl;
            //std::cout << k << ": " << areas[largest_contour_index] << std::endl;
            if(areas[largest_contour_index] >= min_area && areas[largest_contour_index] <= max_area){
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
                //std::cout << "size: " << polygonGroup.size() << std::endl;
//                std::cout << "PolygonGroup" << std::endl;
//                for(int z=0; z < polygonGroup.size(); z++){
//                    std::cout << "Polygon " << z << ": " << polygonGroup[z].polygon.points[z] << polygonGroup[z].polygon.points[1] << polygonGroup[z].polygon.points[2] << polygonGroup[z].polygon.points[3] << std::endl;
//                }

            }
            //std::cout << areas.size() << std::endl;
            //std::cout << k << ": " << areas[largest_contour_index] << "  largest index:" << largest_contour_index << std::endl;
            areas[largest_contour_index] = -1;
            largest_area = 0;

        }
        if(image_perception_pub.getNumSubscribers() > 0)
        {
            hector_perception_msgs::PerceptionDataArray polygonPerceptionArray;
            polygonPerceptionArray.header.stamp = ros::Time::now();
            polygonPerceptionArray.perceptionType = "motion";
            polygonPerceptionArray.perceptionList = polygonGroup;
            image_perception_pub.publish(polygonPerceptionArray);
        }
    }

    //Show results
//    cv::imshow("Binary Image", fgimg);
//    cv::imshow("Motion detected Image", frame);
//    cv::waitKey(0);

    sensor_msgs::CameraInfo::Ptr info;
    info.reset(new sensor_msgs::CameraInfo());
    info->header = img->header;

    if(image_background_subtracted_pub_.getNumSubscribers() > 0)
    {
      cv_bridge::CvImage cvImg;
      fgimg_orig.copyTo(cvImg.image);
      cvImg.header = img->header;
      cvImg.encoding = sensor_msgs::image_encodings::MONO8;
      image_background_subtracted_pub_.publish(cvImg.toImageMsg(), info);
    }

    if(image_motion_pub_.getNumSubscribers() > 0)
    {
      cv_bridge::CvImage cvImg;
      fgimg.copyTo(cvImg.image);
      cvImg.header = img->header;
      cvImg.encoding = sensor_msgs::image_encodings::MONO8;
      image_motion_pub_.publish(cvImg.toImageMsg(), info);
    }

    if(image_detected_pub_.getNumSubscribers() > 0)
    {
      cv_bridge::CvImage cvImg;
      frame.copyTo(cvImg.image);
      cvImg.header = img->header;
      cvImg.encoding = sensor_msgs::image_encodings::BGR8;
      image_detected_pub_.publish(cvImg.toImageMsg(), info);
    }
}

void MotionDetection::dynRecParamCallback(MotionDetectionConfig &config, uint32_t level)
{
  motion_detect_threshold_ = config.motion_detect_threshold;
  min_percept_size = config.motion_detect_min_percept_size;
  max_percept_size = config.motion_detect_max_percept_size;
  min_density = config.motion_detect_min_density;
  percept_class_id_ = config.percept_class_id;

  bg.set ("nmixtures", 3);
  min_area = config.motion_detect_min_area;
  max_area = config.motion_detect_max_area;
  detectionLimit = config.motion_detect_detectionLimit;

  erosion_iterations = config.motion_detect_erosion;
  dilation_iterations = config.motion_detect_dilation;
}

int main(int argc, char **argv)
{
  //cv::namedWindow("view");
  //cv::startWindowThread();
  ros::init(argc, argv, "motion_detection");
  MotionDetection md;
  ros::spin();

  //cv::destroyWindow("view");
}

