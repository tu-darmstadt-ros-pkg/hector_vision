#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>

#include <sensor_msgs/PointCloud2.h>
#include <hector_worldmodel_msgs/PosePercept.h>

#include <hector_soft_obstacle_detection/soft_obstacle_detection.h>

SoftObstacleDetection::SoftObstacleDetection()
{
  ros::NodeHandle nh;

  unit_scale = 100.0;
  border_size = 10.0;

  // subscribe topics
  laser_scan_sub_ = nh.subscribe("/laser1/scan", 1, &SoftObstacleDetection::laserScanCallback, this);

  // advertise topics
  veil_percept_pub_ = nh.advertise<hector_worldmodel_msgs::PosePercept>("veil_percept", 20);
  veil_percept_pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("veil_percept_pose", 20);
  veil_point_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("veil_point_cloud", 5);

  // dynamic reconfigure
  dyn_rec_server_.setCallback(boost::bind(&SoftObstacleDetection::dynRecParamCallback, this, _1, _2));

  update_timer = nh.createTimer(ros::Duration(0.01), &SoftObstacleDetection::update, this);
}

SoftObstacleDetection::~SoftObstacleDetection()
{
}

void SoftObstacleDetection::transformScanToImage(const sensor_msgs::LaserScanConstPtr scan, cv::Mat& img, cv::Point2i& scan_center) const
{
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::min();
  float max_y = std::numeric_limits<float>::min();

  std::vector<cv::Point2f> points;

  // transform all points from polar coordinates into image coordinates
  for (unsigned int i = 0; i < scan->ranges.size(); i++)
  {
    if (!finite(scan->ranges[i]))
      continue;

    cv::Point2f p;
    float angle = scan->angle_max - scan->angle_increment*i;
    p.x = cos(angle) * scan->ranges[i]*unit_scale;
    p.y = sin(angle) * scan->ranges[i]*unit_scale;

    points.push_back(p);

    min_x = std::min(min_x, p.x);
    min_y = std::min(min_y, p.y);
    max_x = std::max(max_x, p.x);
    max_y = std::max(max_y, p.y);
  }

  if (points.empty())
    return;

  // generate image
  cv::Size img_size(max_x-min_x + 2*border_size, max_y-min_y + 2*border_size);

  if (img_size.width <= 2*border_size || img_size.height <= 2*border_size)
    return;

  scan_center.x = border_size - min_x;
  scan_center.y = border_size - min_y;

  img = cv::Mat(img_size, CV_8UC1);
  img = cv::Scalar::all(0);

  for (std::vector<cv::Point2f>::const_iterator itr = points.begin(); itr != points.end(); itr++)
    img.at<uchar>(std::floor(itr->y + scan_center.y), std::floor(itr->x + scan_center.x)) = 255;

  //cv::resize(img, img, cv::Size(400, 400));

#ifdef DEBUG
  cv::imshow("scan", img);
#endif
}

void SoftObstacleDetection::houghTransform(const cv::Mat& img, std::vector<cv::Vec4i>& lines) const
{
  lines.clear();

  // detect lines
  cv::HoughLinesP(img, lines, 1, CV_PI/180, 35, 80, 30);

#ifdef DEBUG
  // draw detected lines
  cv::Mat img_lines(img.size(), CV_8UC3);
  img_lines = cv::Scalar::all(0);
  for (size_t i = 0; i < lines.size(); i++)
  {
    cv::Vec4i line = lines[i];
    cv::line(img_lines, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2, CV_AA);
  }

  cv::imshow("detected_lines", img_lines);
#endif
}

uchar SoftObstacleDetection::getMaxAtLine(const cv::Mat& img, const cv::Point& p1, const cv::Point& p2) const
{
  uchar result = 0;

  cv::LineIterator it(img, p1, p2, 8);
  for(int i = 0; i < it.count; i++, ++it)
  {
    result = std::max(result, img.at<uchar>(it.pos()));

    if (result == 255)
      break;
  }

  return result;
}

void SoftObstacleDetection::getLine(const cv::Mat& img, const cv::Vec4i& line, cv::Mat& out) const
{
  cv::Point p1(line[0], line[1]);
  cv::Point p2(line[2], line[3]);

  std::vector<uchar> signal;
  double r = 3.0;

  cv::Vec2d orth_dir(static_cast<double>(p2.y-p1.y), static_cast<double>(-(p2.x-p1.x)));
  orth_dir = cv::normalize(orth_dir) * r;
  cv::Point o(std::floor(orth_dir(2)+0.5), std::floor(orth_dir(1)+0.5));

  cv::LineIterator it(img, p1, p2, 8);
  for(int i = 0; i < it.count; i++, ++it)
  {
    uchar val1 = getMaxAtLine(img, it.pos()-o, it.pos()+o);
    uchar val2 = getMaxAtLine(img, it.pos()-o-cv::Point(1, 1), it.pos()+o-cv::Point(1, 1));
    signal.push_back(std::max(val1, val2));
  }

  out = cv::Mat(cv::Size(signal.size(), 1), CV_8UC1);
  for (size_t i = 0; i < signal.size(); i++)
    out.at<uchar>(i) = signal[i];
}

void SoftObstacleDetection::edgeDetection(const cv::Mat& signal, std::vector<double>& edges, std::vector<double>& centers) const
{
  //std::vector<double> edges; // pos = abs(c), edge direction = sign(c)
  //std::vector<double> centers; // pos = abs(c), val = sign(c)

  cv::Mat sig(signal.size(), CV_8UC1);
  signal.copyTo(sig);

  // detect edges
  unsigned int edge_dist = 0;
  std::vector<unsigned int> edge_distances;

  // run edge detection
  unsigned int last_val = signal.at<uchar>(0);
  for (int i = 1; i < signal.cols; i++)
  {
    edge_dist++;

    unsigned int current_val = static_cast<int>(signal.at<uchar>(i));
    int d = current_val - last_val;

    if (std::abs(d) > 150) // positiv distances edges
    {
      int sign = d < 0 ? -1 : 1;

      edges.push_back(static_cast<double>(sign * (i-0.5)));

      if (sign < 0 || edge_dist > min_hole_size_)
      {
        edge_distances.push_back(edge_dist);
        centers.push_back(static_cast<double>(-sign) * (static_cast<double>(i-0.5) - 0.5*static_cast<double>(edge_dist)));

        edge_dist = 0;
      }
      else // ignore small holes
      {
        for (int j = i - edge_dist; j < i; j++)
          sig.at<uchar>(j) = 255;

        if (!edge_distances.empty())
        {
          edge_dist += edge_distances[edge_distances.size()-1];
          edge_distances.pop_back();
          edges.pop_back();
          centers.pop_back();
        }
        edges.pop_back();
      }
    }

    last_val = current_val;
  }

  edge_distances.push_back(edge_dist);
  centers.push_back((last_val == 0 ? -1.0 : 1.0) * (static_cast<double>(signal.cols-0.5) - 0.5*static_cast<double>(edge_dist)));

  cv::resize(sig, sig, cv::Size(sig.cols*5, 30));
  cv::imshow("test1", sig);
}

bool SoftObstacleDetection::checkSegmentsMatching(const std::vector<double>& edges, const std::vector<double>& centers, double veil_segment_size, double min_segments, double max_segments) const
{
  if (edges.empty() || centers.empty())
    return false;

  std::vector<double> seg_sizes;
  std::vector<double> seg_dists;

  double mse = 0.0;
  double size_mean = 0.0;
  double size_var = 0.0;
  double dist_mean = 0.0;
  double dist_var = 0.0;

  double last_edge = std::abs(edges[0]);
  for (size_t i = 1; i < edges.size()-1; i++)
  {
    double size = std::abs(edges[i]) - last_edge;
    if (edges[i] < 0)
    {
      double e = veil_segment_size - size;

      mse += e*e;
      size_mean += size;

      seg_sizes.push_back(size);
    }
    else
    {
      dist_mean += size;
      seg_dists.push_back(size);
    }
    last_edge = std::abs(edges[i]);
  }

  if (min_segments > seg_sizes.size() || seg_sizes.size() > max_segments)
    return false;

  mse /= static_cast<double>(seg_sizes.size());
  size_mean /= static_cast<double>(seg_sizes.size());
  dist_mean /= static_cast<double>(seg_dists.size());

  if (size_mean/dist_mean < size_dist_ratio_*0.9 || size_mean/dist_mean > size_dist_ratio_*1.1)
    return false;

  for (std::vector<double>::const_iterator itr = seg_sizes.begin(); itr != seg_sizes.end(); itr++)
    size_var += (*itr - size_mean)*(*itr - size_mean);
  size_var /= static_cast<double>(seg_sizes.size());

  for (std::vector<double>::const_iterator itr = seg_dists.begin(); itr != seg_dists.end(); itr++)
    dist_var += (*itr - dist_mean)*(*itr - dist_mean);
  dist_var /= static_cast<double>(seg_dists.size());

#ifdef DEBUG
  ROS_INFO("MSE: %f, size_mean: %f, size_var: %f, dist_mean: %f, dist_var: %f", mse, size_mean, size_var, dist_mean, dist_var);
#endif

  return mse < max_segment_size_mse_ && size_var < max_segment_size_var_ && dist_var < max_segment_dist_var_;
}

double SoftObstacleDetection::evalModel(const std::vector<double>& edges, const std::vector<double>& centers, double lambda) const
{
  if (centers.size() < 2)
    return -1.0;

  double f = (2.0*M_PI)/lambda;

  double mse = 0.0;

  double bias = std::abs(edges[0]) - (edges[0] > 0.0 ? 0.0 : 0.5*lambda);

  double y = centers[0] < 0.0 ? 1.0 : -1.0;
  for (size_t i = 0; i < centers.size(); i++)
  {
    y *= -1.0;
    double x = f*(std::abs(centers[i])-bias);
    double e = y - sin(x);
    mse += e*e;
  }

  y = edges[0] < 0.0 ? 1.0 : -1.0;
  for (size_t i = 0; i < edges.size(); i++)
  {
    y *= -1.0;
    double x = f*(std::abs(edges[i])-bias);
    double e = y - cos(x);
    mse += e*e;
  }

  return mse/static_cast<double>(centers.size()+edges.size());
}

bool SoftObstacleDetection::checkFrequencyMatching(const std::vector<double>& edges, const std::vector<double>& centers, double lambda, double min_segments, double max_segments) const
{
  if (edges.empty() || centers.empty())
    return false;

  if (min_segments > centers.size()/2 || centers.size()/2 > max_segments)
    return false;

//  edges.clear();
//  centers.clear();

//  centers.push_back(-5);
//  edges.push_back(10);
//  centers.push_back(15);
//  edges.push_back(-20);
//  centers.push_back(-25);
//  edges.push_back(30);
//  centers.push_back(35);
//  edges.push_back(-40);
//  centers.push_back(-45);

//  double mse = evalModel(edges, centers, 20.0);

  double mse = evalModel(edges, centers, lambda);

//  // get average frequency
//  mean /= static_cast<double>(edge_distances.size());

//  // get variance
//  var = 0.0;
//  for (std::vector<unsigned int>::const_iterator itr = edge_distances.begin(); itr != edge_distances.end(); itr++)
//  {
//    double d = (static_cast<double>(*itr) - mean);
//    var += d*d;
//  }
//  var /= static_cast<double>(edge_distances.size());


//  std::string out;
//  for (int i = 0; i < edges.cols; i++)
//    out += boost::lexical_cast<std::string>(static_cast<int>(edges.at<char>(i))) + std::string(" ");
//  ROS_INFO("%s", out.c_str());

#ifdef DEBUG
  std::string out("MSE: ");
  out += boost::lexical_cast<std::string>(mse);
  out += "\nCenters: ";
  for (size_t i = 0; i < centers.size(); i++)
    out += boost::lexical_cast<std::string>(centers[i]) + std::string(" ");
  out += "\nEdges: ";
  for (size_t i = 0; i < edges.size(); i++)
    out += boost::lexical_cast<std::string>(edges[i]) + std::string(" ");

  ROS_INFO("%s", out.c_str());
  //ROS_WARN("Mean: %f, Var: %f (Std: %f)", mean, var, sqrt(var));
#endif

  return mse < max_frequency_mse_;



//  f.clear();

//  cv::Mat padded;
//  int n = cv::getOptimalDFTSize(signal.cols);
//  cv::copyMakeBorder(signal, padded, 0, 0, 0, n - signal.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

//  cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
//  cv::Mat complexI;
//  cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

//  cv::dft(complexI, complexI);            // this way the result may fit in the source matrix

//  // compute the magnitude and switch to logarithmic scale
//  // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
//  cv::split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
//  cv::magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
//  cv::Mat magI = planes[0];

//  magI += cv::Scalar::all(1);                    // switch to logarithmic scale
//  cv::log(magI, magI);

//  // crop the spectrum, if it has an odd number of rows or columns
//  //magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

////  // rearrange the quadrants of Fourier image  so that the origin is at the image center
////  int cx = magI.cols/2;

////  cv::Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
////  cv::Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
////  cv::Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
////  cv::Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

////  cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
////  q0.copyTo(tmp);
////  q3.copyTo(q0);
////  tmp.copyTo(q3);

////  q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
////  q2.copyTo(q1);
////  tmp.copyTo(q2);

//  //cv::normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
//  //                                        // viewable image form (float between values 0 and 1).

//  //imshow("Input Image"       , I   );    // Show the result

//  std::string out;
//  for (int i = 0; i < magI.cols; i++)
//    out += boost::lexical_cast<std::string>(magI.at<float>(i, 1)) + std::string(" ");
//  ROS_INFO("%s", out.c_str());

//  cv::resize(magI, magI, cv::Size(magI.cols*5, 30));
//  cv::imshow("test2", magI);

//  cv::Mat sig;
//  signal.copyTo(sig);
//  cv::resize(sig, sig, cv::Size(sig.cols*5, 30));
//  cv::imshow("test1", sig);
}

void SoftObstacleDetection::lineToPointCloud(const cv::Vec4i& line, const cv::Point2i& scan_center, pcl::PointCloud<pcl::PointXYZ>& out) const
{
  out.clear();

  int x1 = line[0];
  int y1 = line[1];
  int x2 = line[2];
  int y2 = line[3];

  // Bresenham's line algorithm
  const bool steep = (std::abs(y2 - y1) > std::abs(x2 - x1));
  if (steep)
  {
    std::swap(x1, y1);
    std::swap(x2, y2);
  }

  if (x1 > x2)
  {
    std::swap(x1, x2);
    std::swap(y1, y2);
  }

  const int dx = x2 - x1;
  const int dy = std::abs(y2 - y1);

  int error = dx / 2;
  const int ystep = (y1 < y2) ? 1 : -1;

  int y = y1;
  for (int x = x1; x < x2; x++)
  {
    if (steep)
      out.push_back(pcl::PointXYZ(static_cast<double>(y), static_cast<double>(x), 0.0));
    else
      out.push_back(pcl::PointXYZ(static_cast<double>(x), static_cast<double>(y), 0.0));

    error -= dy;
    if (error < 0)
    {
      y += ystep;
      error += dx;
    }
  }
}

void SoftObstacleDetection::update(const ros::TimerEvent& /*event*/)
{
  if (!last_scan)
    return;

  std_msgs::Header header = last_scan->header;

  cv::Mat img_scan;
  cv::Point2i scan_center;
  transformScanToImage(last_scan, img_scan, scan_center);

#ifdef DEBUG
  cv::Mat result;
  cv::cvtColor(img_scan, result, CV_GRAY2RGB);
#endif

  last_scan.reset();

  cv::Mat img_scan_filtered;
  cv::Mat kernel_dil = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3.0, 3.0));
  cv::dilate(img_scan, img_scan, kernel_dil);

  std::vector<cv::Vec4i> lines;
  houghTransform(img_scan, lines);


  //cv::erode(img_scan, img_scan, kernel_dil);

  //cv::dilate(img_scan, img_scan, kernel_dil);
  //cv::imshow("test2", img_scan);

  if (lines.empty())
    return;

  // get signals of each detected line
  std::vector<cv::Mat> signals_;
  for (std::vector<cv::Vec4i>::const_iterator itr = lines.begin(); itr != lines.end(); itr++)
  {
    const cv::Vec4i& line = *itr;
    cv::Point p1(line[0], line[1]);
    cv::Point p2(line[2], line[3]);

    double length_sq = (p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y);

    if (length_sq > max_curtain_length_sq_)
      continue;

    cv::Mat signal;
    getLine(img_scan, line, signal);
    signals_.push_back(signal);

    std::vector<double> edges; // pos = abs(c), edge direction = sign(c)
    std::vector<double> centers; // pos = abs(c), val = sign(c)

    edgeDetection(signal, edges, centers);

    // detect based on thresholds
    //if (min_frequency_ <= f && f <= max_frequency_ && var <= max_var_ /*&&  mse <= max_mse_*/)
    //if (checkFrequencyMatching(edges, centers, 2.0*veil_segment_size_, min_segments_, max_segments_))
    if (checkSegmentsMatching(edges, centers, veil_segment_size_, min_segments_, max_segments_))
    {
#ifdef DEBUG
      ROS_INFO("Detected");
#endif

      if (tf_listener.canTransform("/map", "/laser1_frame", header.stamp))
      {
        // generate pose of soft obstacle
        geometry_msgs::PoseStamped veil_pose;
        veil_pose.header = header;
        veil_pose.pose.position.x =  static_cast<float>((p1.x + p2.x)/2 - scan_center.x) / unit_scale;
        veil_pose.pose.position.y = -static_cast<float>((p1.y + p2.y)/2 - scan_center.y) / unit_scale;
        veil_pose.pose.position.z = 0.0;

        double dx =  ((p2.x - p1.x));
        double dy = -((p2.y - p1.y));
        veil_pose.pose.orientation = tf::createQuaternionMsgFromYaw(atan2(dy, dx) + M_PI_2);

        tf_listener.transformPose("/map", veil_pose, veil_pose);

        // publish veil pose percept
        hector_worldmodel_msgs::PosePercept veil_percept;
        veil_percept.header = veil_pose.header;
        veil_percept.pose.pose = veil_pose.pose;
        veil_percept.info.class_id = "soft_obstacle";
        veil_percept.info.class_support = 1.0;

        veil_percept_pub_.publish(veil_percept);
        veil_percept_pose_pub_.publish(veil_pose);

        // generate line as point cloud
        pcl::PointCloud<pcl::PointXYZ> pc;
        lineToPointCloud(line, scan_center, pc);

        for (size_t i = 0; i < pc.size(); i++)
        {
          pcl::PointXYZ& p = pc.at(i);
          p.x =  (p.x-static_cast<double>(scan_center.x)) / unit_scale;
          p.y = -(p.y-static_cast<double>(scan_center.y)) / unit_scale;
          p.z = 0.0;
        }

        // publish point cloud
        sensor_msgs::PointCloud2 pc_msg;
        pcl::toROSMsg(pc, pc_msg);
        pc_msg.header = header;
        pcl_ros::transformPointCloud("/map", pc_msg, pc_msg, tf_listener);
        veil_point_cloud_pub_.publish(pc_msg);
      }

#ifdef DEBUG
      cv::line(result, p1, p2, cv::Scalar(0, 255, 0), 2, CV_AA);
      cv::imshow("result", result);
      //cv::waitKey();
#endif
    }
  }

#ifdef DEBUG
  cv::imshow("result", result);
  //ROS_INFO("--------------------------------------------------------------");
#endif
}

void SoftObstacleDetection::laserScanCallback(const sensor_msgs::LaserScanConstPtr& scan)
{
  last_scan = scan;
}

void SoftObstacleDetection::dynRecParamCallback(SoftObstacleDetectionConfig& config, uint32_t /*level*/)
{
  min_hole_size_ = config.min_hole_size;
  max_curtain_length_sq_ = config.max_curtain_length*config.max_curtain_length*unit_scale*unit_scale;
  min_frequency_ = config.min_frequency;
  max_frequency_ = config.max_frequency;
  veil_segment_size_ = config.veil_segment_size*unit_scale;
  min_segments_ = config.min_segments;
  max_segments_ = config.max_segments;
  max_segment_size_mse_ = config.max_segment_size_mse;
  max_segment_size_var_ = config.max_segment_size_std*config.max_segment_size_std;
  max_segment_dist_var_ = config.max_segment_dist_std*config.max_segment_dist_std;
  size_dist_ratio_ = config.size_dist_ratio;
  max_frequency_mse_ = config.max_frequency_mse;
  percept_class_id_ = config.percept_class_id;
}

int main(int argc, char **argv)
{
#ifdef DEBUG
  cv::namedWindow("scan");
  cv::namedWindow("detected_lines");
  cv::namedWindow("result");
  cv::namedWindow("test1");
  cv::namedWindow("test2");
  cv::startWindowThread();
#endif

  ros::init(argc, argv, "soft_obstacle_detection");
  SoftObstacleDetection sod;
  ros::spin();

#ifdef DEBUG
  cv::destroyWindow("scan");
  cv::destroyWindow("detected_lines");
  cv::destroyWindow("result");
  cv::destroyWindow("test1");
  cv::destroyWindow("test2");
#endif

  return 0;
}

