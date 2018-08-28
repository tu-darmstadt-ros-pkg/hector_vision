//
// Created by Stefan Fabian on 11.06.18.
//

#include "hector_pipe_detection/circle_detection.h"

#include <hector_vision_algorithms/color_edges.h>
#include <hector_vision_algorithms/thesholding.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <nodelet/nodelet.h>

namespace hector_pipe_detection
{

DebugInfo::DebugInfo( std::shared_ptr<image_transport::ImageTransport> transport )
{
  edge_image_pub_ = transport->advertise( "edge_image", 1 );
  filtered_contours_pub_ = transport->advertise( "filtered_contours", 1 );
  sub_edge_image_pub_ = transport->advertise( "sub_edge_image", 1 );
  sub_image_contours_pub_ = transport->advertise( "sub_image_contours", 1 );
  detection_pub_ = transport->advertise( "detection", 1 );
}

void DebugInfo::publishEdgeImage( const cv::Mat &edge_image )
{
  edge_image_pub_.publish( cv_bridge::CvImage( std_msgs::Header(), "8UC1", edge_image ).toImageMsg());
}

void DebugInfo::publishFilteredContours( const cv::Mat &image, std::vector<std::vector<cv::Point>> contours )
{
  cv::Mat out_image = image.clone();
  cv::drawContours( out_image, contours, -1, cv::Scalar( 0, 255, 0 ), 2 );
  filtered_contours_pub_.publish( cv_bridge::CvImage( std_msgs::Header(), "rgb8", out_image ).toImageMsg());
}

void DebugInfo::publishSubEdgeImage( const cv::Mat &sub_edge_image )
{
  sub_edge_image_pub_.publish( cv_bridge::CvImage( std_msgs::Header(), "8UC1", sub_edge_image ).toImageMsg());
}

void DebugInfo::publishSubImageContours( const cv::Mat &sub_image, std::vector<std::vector<cv::Point>> contours,
                                         const cv::Point &offset )
{
  cv::Mat out_image = sub_image.clone();
  cv::drawContours( out_image, contours, -1, cv::Scalar( 0, 255, 0 ), 2, cv::LINE_8, cv::noArray(), 0x7fffffff,
                    offset );
  sub_image_contours_pub_.publish( cv_bridge::CvImage( std_msgs::Header(), "rgb8", out_image ).toImageMsg());
}

void DebugInfo::publishDetection( const cv::Mat &image, const cv::Point2d &center, double radius )
{
  ROS_DEBUG( "Center: %f, %f; Radius: %f", center.x, center.y, radius );
  cv::Mat out_image = image.clone();
  cv::circle( out_image, center, (int) (radius + 0.5), cv::Scalar( 255, 0, 0 ), 2 );
  detection_pub_.publish( cv_bridge::CvImage( std_msgs::Header(), "rgb8", out_image ).toImageMsg());
}


// =======================================================================================
// ================================== Detection Methods ==================================
// =======================================================================================


inline double square( double x )
{
  return x * x;
}


bool findCircles( const std::vector<std::vector<cv::Point>> &contours, int downsample_passes,
                  std::vector<std::vector<cv::Point>> &filtered_contours, std::vector<int> &largest_group,
                  int &largest )
{
  std::vector<double> areas;
  std::vector<cv::Point2d> centers;
  unsigned int multiplier = 1U << downsample_passes;
  for ( int i = 0; i < contours.size(); ++i )
  {
    std::vector<cv::Point> approx;
    cv::convexHull( contours[i], approx );
    cv::approxPolyDP( approx, approx, 0.005 * cv::arcLength( approx, true ), true );
    double area = cv::contourArea( approx ) * square(multiplier);
    if ( area < 500 || cv::abs( square( multiplier * cv::arcLength( contours[i], true )) / (4 * M_PI * area) - 1 ) > 0.2 )
    {
      continue;
    }

    cv::Moments moments = cv::moments( approx );
    cv::Point2d center( moments.m10 / moments.m00, moments.m01 / moments.m00 );
    int j;
    for ( j = 0; j < centers.size(); ++j )
    {
      if ( centers[j].x > center.x )
      {
        centers.insert( centers.begin() + j, center );
        filtered_contours.insert( filtered_contours.begin() + j, contours[i] );
        areas.insert( areas.begin() + j, area );
        break;
      }
    }
    if ( j == centers.size())
    {
      centers.push_back( center );
      filtered_contours.push_back( contours[i] );
      areas.push_back( area );
    }
  }

  if ( filtered_contours.empty()) return false;

  int largest_group_start = 0;
  int largest_group_length = 0;
  double largest_group_area = areas[0];
  double group_area = areas[0];
  int start = 0;
  for ( int i = 1; i < centers.size(); ++i )
  {
    if ( square( centers[i].x - centers[i - 1].x ) + square( centers[i].y - centers[i - 1].y ) < 400 )
    {
      if ( areas[i] > group_area )
      {
        group_area = areas[i];
      }
      continue;
    }
    if ( i - start > largest_group_length || (i - start == largest_group_length && group_area > largest_group_area))
    {
      largest_group_start = start;
      largest_group_length = i - start;
      largest_group_area = group_area;
    }
    start = i;
    group_area = areas[i];
  }
  if ( centers.size() - start > largest_group_length ||
       (centers.size() - start == largest_group_length && group_area > largest_group_area))
  {
    largest_group_start = start;
    largest_group_length = centers.size() - start;
  }

  // Store indices of largest group and remove octagon if found.
  largest = 0;
  int second_largest = 0;
  double max_arc = 0;
  double second_max_arc = 0;
  for ( int i = largest_group_start; i < largest_group_start + largest_group_length; ++i )
  {
    double arc = cv::arcLength( filtered_contours[i], true );
    if ( arc > max_arc )
    {
      second_max_arc = max_arc;
      second_largest = largest;
      max_arc = arc;
      largest = i;
    }
    else if ( arc > second_max_arc )
    {
      second_max_arc = arc;
      second_largest = i;
    }
  }
  for ( int i = largest_group_start; i < largest_group_start + largest_group_length; ++i )
  {
    if ( largest == i && filtered_contours.size() == 8 )
    {
      largest = second_largest;
      continue;
    }
    largest_group.push_back( i );
  }
  return true;
}

double euclideanDistance( cv::Point2d a, cv::Point b )
{
  double xdiff = a.x - b.x;
  double ydiff = a.y - b.y;
  return cv::sqrt( xdiff * xdiff + ydiff * ydiff );
}

void circlify( std::vector<cv::Point> contour, cv::Point2d &center, double &radius )
{
  cv::Moments moments = cv::moments( contour );
  center = cv::Point2d( moments.m10 / moments.m00, moments.m01 / moments.m00 );
  double distances[contour.size()];
  double mean = 0;
  for ( int i = 0; i < contour.size(); ++i )
  {
    distances[i] = euclideanDistance( center, contour[i] );
    mean += distances[i];
  }
  mean /= contour.size();
  double stddev = 0;
  for ( int i = 0; i < contour.size(); ++i )
  {
    stddev += (distances[i] - mean) * (distances[i] - mean);
  }
  stddev = cv::sqrt( stddev / (contour.size() - 1));

  std::vector<int> filtered_contour_index;
  std::vector<cv::Point> filtered_contour;
  for ( int i = 0; i < contour.size(); ++i )
  {
    if ( cv::abs( distances[i] - mean ) > stddev ) continue;
    filtered_contour_index.push_back( i );
    filtered_contour.push_back( contour[i] );
  }

  moments = cv::moments( filtered_contour );
  center = cv::Point2d( moments.m10 / moments.m00, moments.m01 / moments.m00 );
  for ( int i = 0; i < contour.size(); ++i )
  {
    distances[i] = euclideanDistance( center, contour[i] );
  }
  mean = 0;
  for ( int i = 0; i < filtered_contour.size(); ++i )
  {
    mean += distances[filtered_contour_index[i]];
  }
  mean /= filtered_contour.size();
  stddev = 0;
  for ( int i = 0; i < filtered_contour.size(); ++i )
  {
    stddev += (distances[filtered_contour_index[i]] - mean) * (distances[filtered_contour_index[i]] - mean);
  }
  stddev = cv::sqrt( stddev / (filtered_contour.size() - 1));

  filtered_contour.clear();
  for ( int i = 0; i < contour.size(); ++i )
  {
    if ( cv::abs( distances[i] - mean ) > 1.5 * stddev ) continue;
    filtered_contour.push_back( contour[i] );
  }

  moments = cv::moments( filtered_contour );
  center = cv::Point2d( moments.m10 / moments.m00, moments.m01 / moments.m00 );
  mean = 0;
  for ( const auto &point : filtered_contour )
  {
    mean += euclideanDistance( center, point );
  }

  radius = mean / filtered_contour.size();
}

bool getCircle(const cv::Point &a, const cv::Point &b, const cv::Point &c, double &x, double &y, double &r)
{
  if (b.x == a.x) {
    if (b.x == c.x) return false;
    return getCircle(a, c, b, x, y, r);
  }
  double m_a = (double)(b.y - a.y) / (b.x - a.x);
  double m_b = (double)(c.y - b.y) / (c.x - b.x);
  x = (m_a * m_b * (a.y - c.y) + m_b * (a.x + b.x) - m_a * (b.x + c.x)) / 2 * (m_b - m_a);
  if (m_a != 0)
  {
    y = -1 / m_a * (x - (a.x + b.x) / 2.0) + (a.y + b.y) / 2.0;
  }
  else
  {
    y = -1 / m_b * (x - (b.x + c.x) / 2.0) + (b.y + c.y) / 2.0;
  }
  r = cv::sqrt(square(a.x - x) + square(a.y - y));
  return true;
}

void findCircle( const cv::Mat &edges, std::vector<std::vector<cv::Point> > contours, unsigned int scaling_factor, double &cx, double &cy, double &cr)
{
  unsigned int max_iterations = 50;
  double best_ratio = 0;
  cx = NAN;
  cy = NAN;
  cr = NAN;
  for (auto it = contours.begin(); it != contours.end(); ++it)
  {
    if (it->size() < 20) continue;
    for (int i = 0; i < max_iterations; ++i)
    {
      size_t a, b, c;
      do
      {
        a = random() % it->size();
        b = random() % it->size();
        c = random() % it->size();
      } while ( a == b || a == c || b == c );
      double x, y, r;
      getCircle((*it)[a], (*it)[b], (*it)[c], x, y, r);
      if (r * scaling_factor < 40) continue;
      unsigned int inliers = 0;
      unsigned int count = 0;
//      int row_start = y - r - 1;
//      if (row_start < 0) row_start = 0;
//      int row_end = y + r + 2.5;
//      if (row_end > edges.rows) row_end = edges.rows;
//      int col_start = x - r - 1;
//      if (col_start < 0) col_start = 0;
//      int col_end = x + r + 2.5;
//      if (col_end > edges.cols) col_end = edges.cols;
//      for (int row = row_start; row < row_end; ++row)
//      {
//        for (int col = col_start; col < col_end; ++col)
//        {
//          if (cv::abs(cv::sqrt(square(row - y) + square(col - x)) - r) > 1) continue;
//          count += 1;
//          if (edges.at<unsigned char>(row, col) != 0) ++inliers;
//        }
//      }
    for (double angle = 0; angle < 2 * M_PI; angle += 0.05)
    {
      int x = r * cos(angle) + 0.5;
      int y = r * sin(angle) + 0.5;
      if (x >= 0 && x < edges.cols && y >= 0 && y < edges.rows)
      {
        ++count;
        if (edges.at<unsigned char>(y, x) != 0)
        {
          ++inliers;
        }
      }
    }
      if (inliers > 10 && inliers / (double)count > best_ratio)
      {
        best_ratio = inliers / (double) count;
        cx = x;
        cy = y;
        cr = r;
      }
    }
  }
}

bool findOuterCircle( const cv::Mat &image, int downsample_passes, cv::Point2d &center, double &radius,
                      std::shared_ptr<DebugInfo> debug_info )
{
  cv::Mat image_downsampled = image;
  int passes = downsample_passes;
  unsigned int scaling = 1U << downsample_passes;
  while ( passes-- > 0 ) cv::pyrDown( image_downsampled, image_downsampled );

  cv::Mat edges = hector_vision_algorithms::calculateColorEdges( image_downsampled );
  double upper, lower;
  hector_vision_algorithms::calculateThresholds( edges, upper, lower );
  edges = hector_vision_algorithms::threshold( edges, lower );
  // Exclude Pen
  int x_pen = 360;
  int y_pen = 240;
  int height_pen = 40;
  for (int row = y_pen / scaling; row < (y_pen + height_pen) / scaling; ++row)
  {
    for ( int col = x_pen / scaling; col < edges.cols; ++col )
    {
      edges.at<unsigned char>(row, col) = 0;
    }
  }
  if ( debug_info != nullptr )
  {
    debug_info->publishEdgeImage( edges );
  }
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours( edges, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
  if ( debug_info != nullptr )
  {
    debug_info->publishFilteredContours( image_downsampled, contours );
  }
  double cx, cy, cr;
  findCircle(edges, contours, scaling, cx, cy, cr);
  if (isnanl(cr) || cr < 40) return false;
  contours.clear();

  if ( downsample_passes != 0 )
  {
    cv::Rect rect( cx - cr, cy - cr, 2 * cr, 2 * cr );
    rect.x *= scaling;
    rect.y *= scaling;
    rect.width *= scaling;
    rect.height *= scaling;
    const int margin = 10;
    rect.x -= margin;
    rect.y -= margin;
    rect.width += 2 * margin;
    rect.height += 2 * margin;
    if ( rect.x < 0 ) rect.x = 0;
    if ( rect.y < 0 ) rect.y = 0;
    if ( rect.x + rect.width > image.cols ) rect.width = image.cols - rect.x;
    if ( rect.y + rect.height > image.rows ) rect.height = image.rows - rect.y;
    cv::Mat sub_image = image( rect );
    std::cout << "SubImage: " << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height << std::endl;

    edges = hector_vision_algorithms::calculateColorEdges( sub_image );
    hector_vision_algorithms::calculateThresholds( edges, upper, lower );
    edges = hector_vision_algorithms::threshold( edges, upper, lower );
    // Exclude Pen
    for ( int row = y_pen - rect.y; row < (y_pen + height_pen) && row < rect.height; ++row )
    {
      for ( int col = x_pen - rect.x; col < edges.cols && col < rect.width; ++col )
      {
        edges.at<unsigned char>( row, col ) = 0;
      }
    }
    std::vector<std::vector<cv::Point>> sub_contours;
    std::cout << "Edges " << edges.rows << ", " << edges.cols << std::endl;
    if ( debug_info != nullptr )
    {
      debug_info->publishSubEdgeImage( edges );
    }
    cv::findContours( edges, sub_contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cv::Point( rect.x, rect.y ));
    if ( debug_info != nullptr )
    {
      debug_info->publishSubImageContours( sub_image, sub_contours, cv::Point( -rect.x, -rect.y ));
    }
    findCircle( edges, sub_contours, 1, cx, cy, cr );
  }
  center = cv::Point2d(cx, cy);
  radius = cr;
  if ( isnanl( center.x ) || isnanl( center.y ) || isnanl( radius ))
    return false;
  if ( debug_info != nullptr )
  {
    debug_info->publishDetection( image, center, radius );
  }
  return true;
}
}
