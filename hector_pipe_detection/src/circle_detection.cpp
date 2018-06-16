//
// Created by Stefan Fabian on 11.06.18.
//

#include "hector_pipe_detection/circle_detection.h"

#include <hector_vision_algorithms/color_edges.h>
#include <hector_vision_algorithms/thesholding.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <nodelet/nodelet.h>
#include <hector_profiling/timer.h>

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

void DebugInfo::publishFilteredContours( const cv::Mat &image, const std::vector<std::vector<cv::Point>> &contours )
{
  cv::Mat out_image = image.clone();
  cv::drawContours( out_image, contours, -1, cv::Scalar( 0, 255, 0 ), 2 );
  filtered_contours_pub_.publish( cv_bridge::CvImage( std_msgs::Header(), "rgb8", out_image ).toImageMsg());
}

void DebugInfo::publishSubEdgeImage( const cv::Mat &sub_edge_image )
{
  sub_edge_image_pub_.publish( cv_bridge::CvImage( std_msgs::Header(), "8UC1", sub_edge_image ).toImageMsg());
}

void DebugInfo::publishSubImageContours( const cv::Mat &sub_image, const std::vector<std::vector<cv::Point>> &contours,
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


using namespace hector_profiling;

bool findOuterCircle( const cv::Mat &image, int downsample_passes, cv::Point2d &center, double &radius,
                      std::shared_ptr<DebugInfo> debug_info )
{
  cv::Mat image_downsampled = image;
  int passes = downsample_passes;
  unsigned int scaling = 1U << downsample_passes;
  while ( passes-- > 0 ) cv::pyrDown( image_downsampled, image_downsampled );

  cv::Mat edges = hector_vision_algorithms::color_edges( image_downsampled );
  double upper, lower;
  hector_vision_algorithms::calculateThresholds( edges, upper, lower );
  edges = hector_vision_algorithms::threshold( edges, lower );
  if ( debug_info != nullptr )
  {
    debug_info->publishEdgeImage( edges );
  }
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours( edges, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
  std::vector<std::vector<cv::Point>> filtered_contours;
  std::vector<int> largest_group;
  int largest;
  if ( !findCircles( contours, downsample_passes, filtered_contours, largest_group, largest ))
    return false;
  if ( debug_info != nullptr )
  {
    debug_info->publishFilteredContours( image_downsampled, filtered_contours );
  }

  if ( downsample_passes != 0 )
  {
    cv::Rect rect = cv::boundingRect( filtered_contours[largest] );
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

    edges = hector_vision_algorithms::color_edges( sub_image );
    hector_vision_algorithms::calculateThresholds( edges, upper, lower );
    edges = hector_vision_algorithms::threshold( edges, upper, lower );
    contours.clear();
    cv::findContours( edges, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point( rect.x, rect.y ));
    if ( debug_info != nullptr )
    {
      debug_info->publishSubEdgeImage( edges );
      debug_info->publishSubImageContours( sub_image, contours, cv::Point( -rect.x, -rect.y ));
    }
  }
  else
  {
    contours = filtered_contours;
  }
  double largest_arc = 0;
  largest = 0;
  for ( int i = 0; i < contours.size(); ++i )
  {
    double arc = cv::arcLength( contours[i], true );
    if ( arc <= largest_arc ) continue;
    largest_arc = arc;
    largest = i;
  }
  std::vector<cv::Point> circle_contour;
  for ( int i = 0; i < contours[largest].size(); ++i )
  {
    circle_contour.push_back( contours[largest][i] );
  }
  circlify( circle_contour, center, radius );
  if ( isnanl( center.x ) || isnanl( center.y ) || isnanl( radius ))
    return false;
  if ( debug_info != nullptr )
  {
    debug_info->publishDetection( image, center, radius );
  }
  return true;
}
}
