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
    image_transport::ImageTransport image_rects_it(p_n);

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
    ROS_INFO("...");
    ROS_DEBUG("PARAMS:");
    ROS_DEBUG("max area: %d", max_area);
    ROS_DEBUG("min area: %d", min_area);
    ROS_DEBUG("detection limit: %d", detectionLimit);
    ROS_DEBUG("closing iterations: %d", closing_iterations);
    ROS_DEBUG("erosion iterations: %d", erosion_iterations);
    ROS_DEBUG("dilation iterations: %d", dilation_iterations);
    ROS_DEBUG("kernel size: %d", ikernelSize);
    ROS_DEBUG("blob morph iterations: %d", blob_morph_iterations);
    ROS_DEBUG("filter by area: %i", params.filterByArea);
    ROS_DEBUG("params.minArea: %f", params.minArea);
    ROS_DEBUG("params.maxArea: %f", params.maxArea);
    ROS_DEBUG("filter by circularity: %i", params.filterByCircularity);
    ROS_DEBUG("params.minCircularity: %f", params.minCircularity);
    ROS_DEBUG("params.maxCircularity: %f", params.maxCircularity);
    image_transport::ImageTransport image_bg_it(p_n);
    image_background_subtracted_pub_ = image_bg_it.advertiseCamera("image_background_subtracted", 10);
    image_rects_pub_ = image_rects_it.advertiseCamera("image_rects", 10);
}

MotionDetection::~MotionDetection() {}

void MotionDetection::imageCallback(const sensor_msgs::ImageConstPtr& img) //, const sensor_msgs::CameraInfoConstPtr& info)
{
    //============INITIAL STEPS================================

    cv_bridge::CvImageConstPtr cv_ptr;
    cv_ptr = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::BGR8);
    cv::Mat img_filtered(cv_ptr->image);

    std::vector < std::vector < cv::Point > >contours;
    cv::Mat frame, fgimg, backgroundImage, fgimg_orig, rects_img, locations;

    img_filtered.copyTo(fgimg);
    img_filtered.copyTo(frame);

    bg.operator()(img_filtered, fgimg);

    for(int i=0; i < closing_iterations; i++){
        cv::morphologyEx(fgimg, fgimg, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));
    }

    fgimg.copyTo(fgimg_orig);   //for debugging/tuning purposes
    bg.getBackgroundImage (backgroundImage);


    //controlable iterations for morphological operations
    for(int i=0; i < erosion_iterations; i++){
        cv::erode (fgimg, fgimg, cv::Mat());
    }
    for(int i=0; i < dilation_iterations; i++){
        cv::dilate (fgimg, fgimg, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10,10))); //alternatively (tuning): other kernels: e.g. cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10,10))
    }

    cv::findContours (fgimg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    //cv::imshow("Binary", fgimg);
    cv::drawContours (frame, contours, -1, cv::Scalar (255, 0, 0), 2);

    //========DETECTION OF LARGEST CONTOURS=====
    int largest_area=0;
    int largest_contour_index=0;
    cv::Rect bounding_rect;
    std::vector<cv::Rect> boundRect( contours.size() );
    std::vector<cv::vector<cv::Point> > contours_poly( contours.size() );
    std::vector<cv::Point2f>center( contours.size() );
    std::vector<float>radius( contours.size() );
    std::vector<double> areas( contours.size() );

    //std::vector<double> rectAreas;

    for( int i = 0; i < contours.size(); i++ )
    {
        double area = cv::contourArea( contours[i] );  //  Find the area of contour
        areas[i] = area;
    }

    std::vector<cv::Rect> rectGroup(detectionLimit);

    std::vector<hector_perception_msgs::PerceptionData> polygonGroup;

    //check frame data
    //ROS_INFO("Pixels width: %d", frame.cols);
    //ROS_INFO("Pixels height:%d", frame.rows);

    //check for number of changes
    cv::findNonZero(fgimg, locations);
    num_of_changes = locations.rows;
    //ROS_INFO("Number of changes: %d", locations.rows);

    if(contours.size() != 0 && num_of_changes < tolerance){

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
                frame.copyTo(rects_img);
                rectGroup.push_back(bounding_rect);    //for Kalman try

            }
            //std::cout << areas.size() << std::endl;
            //std::cout << k << ": " << areas[largest_contour_index] << "  largest index:" << largest_contour_index << std::endl;
            areas[largest_contour_index] = -1;
            largest_area = 0;
        }


        //==========KALMAN APPROACH=======================

        m1 = m1 * reduction;   //reduce value of every pixel
        for(std::vector<cv::Rect>::size_type i=0; i < rectGroup.size(); i++){
            m1(rectGroup[i]) += weight;    //empower pixels in rectangle
        }

        cv::normalize(m1, out, 0, 1, cv::NORM_MINMAX);  //normalize to [0,1]

        cv::threshold(out, out, 0.1, 255, cv::THRESH_BINARY_INV);   //segmentation

        for(int i=0; i < blob_morph_iterations; i++){
            cv::erode(out, out, cv::getStructuringElement(cv::MORPH_RECT, kernelSize));  //dilation
        }

        cv::vector<cv::KeyPoint> keypoints;

//        // Change thresholds
//        params.minThreshold = 10;
//        params.maxThreshold = 200;

        params.filterByArea = true;
        params.filterByCircularity = false;

        // Filter by Convexity
        params.filterByConvexity = false;
        params.minConvexity = 0.087;

        // Filter by Inertia
        params.filterByInertia = false;
        params.minInertiaRatio = 0.01;
        cv::SimpleBlobDetector detector(params);

        params.filterByColor = false;

        //convert
        keypoints.clear();
        out.convertTo(out,CV_8UC1);

        //edge correction
        for(int j=0; j < 3; j++){
            for(int i=0; i < out.cols; i++){
                out.row(0+j).col(i) = 255;
                out.row(out.rows-1-j).col(i) = 255;
            }
            for(int i=0; i < out.rows; i++){
                out.row(i).col(0+j) = 255;
                out.row(i).col(out.cols-1-j) = 255;
            }
        }

        //blob detection
        detector.detect( out, keypoints);
        //std::cout << keypoints.size() << std::endl;

        //cv::drawKeypoints( out, keypoints, out_with_keypoints, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        //cv::drawKeypoints( img_filtered, keypoints, frame, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

        for(std::vector<cv::KeyPoint>::size_type i=0; i != keypoints.size(); ++i){
            cv::Point2f f = cv::Point2f(keypoints[i].pt.x - keypoints[i].size, keypoints[i].pt.y - keypoints[i].size);
            cv::Point2f s = cv::Point2f(keypoints[i].pt.x + keypoints[i].size, keypoints[i].pt.y + keypoints[i].size);
            cv::rectangle( frame, f, s, cv::Scalar( 0, 0, 255 ), 3, 8, 0 );


            //Create a Polygon consisting of Point32 points for publishing
            //std::cout << "BR first: " << bounding_rect.tl() << "BR lasst: " << bounding_rect.br() << std::endl;
            geometry_msgs::Point32 p1, p2, p3, p4;
            p1.x = f.x;            p1.y = f.y;
            p4.x = s.x;            p4.y = s.y;
            p2.x = p4.x;           p2.y = p1.y;
            p3.x = p1.x;           p3.y = p4.y;

            //std::cout << "Size of Rect: " << k << " " << bounding_rect.area() << std::endl;
            //rectAreas.push_back(bounding_rect.area());

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
            perceptionData.percept_name = "motion" + boost::lexical_cast<std::string>(i);
            perceptionData.polygon = polygon;

            polygonGroup.push_back(perceptionData);
            //std::cout << "size: " << polygonGroup.size() << std::endl;
//                std::cout << "PolygonGroup" << std::endl;
//                for(int z=0; z < polygonGroup.size(); z++){
//                    std::cout << "Polygon " << z << ": " << polygonGroup[z].polygon.points[z] << polygonGroup[z].polygon.points[1] << polygonGroup[z].polygon.points[2] << polygonGroup[z].polygon.points[3] << std::endl;
//                }

        }
        shake_count--;
    }
    else {
        if(shake_count >= 10){
            ROS_INFO("Stabilize image, too many changes detected");
            shake_count = 0;
        }
        ROS_DEBUG("Too many motions/Noise in the picture, number of changes too high, skipping frame");
        shake_count++;
    }

    //==========PUBLISHING=======================

    //Show results
//    cv::imshow("Binary Image", fgimg);
//    cv::imshow("Motion detected Image", frame);
//    cv::waitKey(0);

    sensor_msgs::CameraInfo::Ptr info;
    info.reset(new sensor_msgs::CameraInfo());
    info->header = img->header;

    if(image_perception_pub.getNumSubscribers() > 0)
    {
        hector_perception_msgs::PerceptionDataArray polygonPerceptionArray;
        polygonPerceptionArray.header.stamp = ros::Time::now();
        polygonPerceptionArray.perceptionType = "motion";
        polygonPerceptionArray.perceptionList = polygonGroup;
        image_perception_pub.publish(polygonPerceptionArray);
    }

    if(image_background_subtracted_pub_.getNumSubscribers() > 0)    //change name to: image_blob_agglomeration
    {
      cv_bridge::CvImage cvImg;
      //fgimg_orig.copyTo(cvImg.image);
      out_with_keypoints.copyTo(cvImg.image);
      cvImg.header = img->header;
      cvImg.encoding = sensor_msgs::image_encodings::BGR8;
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

//    if(image_rects_pub_.getNumSubscribers() > 0)
//    {
//      cv_bridge::CvImage cvImg;
//      rects_img.copyTo(cvImg.image);
//      cvImg.header = img->header;
//      cvImg.encoding = sensor_msgs::image_encodings::BGR8;
//      image_rects_pub_.publish(cvImg.toImageMsg(), info);
//    }
}

void MotionDetection::dynRecParamCallback(MotionDetectionConfig &config, uint32_t level)
{
//  motion_detect_threshold_ = config.motion_detect_threshold;
//  min_percept_size = config.motion_detect_min_percept_size;
//  max_percept_size = config.motion_detect_max_percept_size;
//  min_density = config.motion_detect_min_density;
//  percept_class_id_ = config.percept_class_id;

  bg.set ("nmixtures", 3);
  min_area = config.motion_detect_min_area;
  max_area = config.motion_detect_max_area;
  detectionLimit = config.motion_detect_detectionLimit;

  erosion_iterations = config.motion_detect_erosion;
  dilation_iterations = config.motion_detect_dilation;
  closing_iterations = config.motion_detect_closing;

  //=============================================
  m1 = cv::Mat(480, 640, CV_32F, double(0));    //initialize empty matrix with frame data
  //std::cout << "m1: " <<  m1.at<double>(0,0) << "  ---- " << m1.rows << std::endl;
  out; //for Kalman
  reduction = config.motion_detect_reduction;
  weight = config.motion_detect_weight;
  out_with_keypoints;

  params;
  // Filter by Area.
  params.minArea = config.motion_detect_min_area_blob;
  params.maxArea = config.motion_detect_max_area_blob;

  // Filter by Circularity
  params.minCircularity = config.motion_detect_min_circularity_blob;
  params.maxCircularity = config.motion_detect_max_circularity_blob;
  ikernelSize = config.motion_detect_kernelSize;
  kernelSize = cv::Size(ikernelSize, ikernelSize);
  blob_morph_iterations = config.motion_detect_blob_morph;
  tolerance = config.motion_detect_tolerance;
  shake_count;
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

