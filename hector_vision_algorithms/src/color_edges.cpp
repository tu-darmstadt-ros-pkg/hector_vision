//
// Created by Stefan Fabian on 15.06.18.
//

#include "hector_vision_algorithms/color_edges.h"

#include <opencv2/imgproc.hpp>

namespace hector_vision_algorithms
{

void getJacobians( const cv::Mat &response_x, const cv::Mat &response_y, cv::Mat &jx, cv::Mat &jy, cv::Mat &jxy )
{
  cv::Mat channels_x[3];
  cv::split( response_x, channels_x );
  jx = channels_x[0].mul( channels_x[0] );
  jx += channels_x[1].mul( channels_x[1] );
  jx += channels_x[2].mul( channels_x[2] );

  cv::Mat channels_y[3];
  cv::split( response_y, channels_y );
  jy = channels_y[0].mul( channels_y[0] );
  jy += channels_y[1].mul( channels_y[1] );
  jy += channels_y[2].mul( channels_y[2] );

  jxy = channels_x[0].mul( channels_y[0] );
  jxy += channels_x[1].mul( channels_y[1] );
  jxy += channels_x[2].mul( channels_y[2] );
}

// Returns true if element at (x1, y1) is smaller than element at (x2, y2)
// If no element exists at (x2, y2) or it is not smaller, it returns false
template<typename T>
inline bool is_smaller( const cv::Mat &mat, int x1, int y1, int x2, int y2 )
{
  if ( x2 < 0 || x2 >= mat.cols ) return false;
  if ( y2 < 0 || y2 >= mat.rows ) return false;
  return mat.at<T>( y1, x1 ) < mat.at<T>( y2, x2 );
}

cv::Mat calculateColorEdges( const cv::Mat &image )
{
  int filter_type = CV_16S;
  cv::Mat x_filter( 3, 3, filter_type, cv::Scalar( 0 ));
  typedef short FilterType;
  x_filter.at<FilterType>( 0, 0 ) = 1;
  x_filter.at<FilterType>( 0, 1 ) = 2;
  x_filter.at<FilterType>( 0, 2 ) = 1;
  x_filter.at<FilterType>( 2, 0 ) = -1;
  x_filter.at<FilterType>( 2, 1 ) = -2;
  x_filter.at<FilterType>( 2, 2 ) = -1;
  cv::Mat y_filter;
  cv::transpose( x_filter, y_filter );

  cv::Mat response_x;
  cv::filter2D( image, response_x, filter_type, x_filter );
  response_x.convertTo( response_x, CV_32F );

  cv::Mat response_y;
  cv::filter2D( image, response_y, filter_type, y_filter );
  response_y.convertTo( response_y, CV_32F );

  cv::Mat jx, jy, jxy;
  getJacobians( response_x, response_y, jx, jy, jxy );

  // compute first (greatest) eigenvalue of 2x2 matrix J'*J.
  // note that the abs() is only needed because some values may be slightly
  // negative due to round-off error.
  cv::Mat d;
  cv::sqrt( cv::abs( jx.mul( jx ) - 2 * jx.mul( jy ) + jy.mul( jy ) + 4 * jxy.mul( jxy )), d );
  cv::Mat e1 = (jx + jy + d) / 2;
  // the 2nd eigenvalue would be:  e2 = (Jx + Jy - D) / 2
  cv::Mat edges;
  cv::sqrt( e1, edges );

  return edges;
  cv::Mat mean, stddev;
  cv::meanStdDev( edges, mean, stddev );
  double high_threshold = mean.at<double>( 0, 0 ) + 1 * stddev.at<double>( 0, 0 );
  cv::Mat out = cv::Mat( edges.rows, edges.cols, CV_8U );
  float min_orientation = 360;
  float max_orientation = 0;

  for ( int row = 0; row < edges.rows; ++row )
  {
    for ( int col = 0; col < edges.cols; ++col )
    {
      // 0 degrees is the positive X-Axis, degrees increment counter-clockwise
      // We move it so that up is 90, down is -90 and flip the left side to the right
      float orientation =
        cv::fastAtan2( -jxy.at<float>( row, col ), e1.at<float>( row, col ) - jy.at<float>( row, col )) - 90;
      if ( orientation > 270 ) orientation -= 360;
      else if ( orientation > 90 ) orientation -= 180;
      if ( orientation > max_orientation ) max_orientation = orientation;
      if ( orientation < min_orientation ) min_orientation = orientation;
      bool zero;
      if ( edges.at<float>( row, col ) < high_threshold )
      {
        zero = true;
      }
      else if ( orientation > 67.5 || orientation <= -67.5 ) // Vertical
      {
        zero = is_smaller<float>( edges, col, row, col, row - 1 ) || is_smaller<float>( edges, col, row, col, row + 1 );
      }
      else if ( orientation > 22.5 ) // Bottom-Left to Top-Right
      {
        zero = is_smaller<float>( edges, col, row, col + 1, row - 1 ) ||
               is_smaller<float>( edges, col, row, col - 1, row + 1 );
      }
      else if ( orientation > -22.5 ) // Horizontal
      {
        zero = is_smaller<float>( edges, col, row, col - 1, row ) || is_smaller<float>( edges, col, row, col + 1, row );
      }
      else // Top-Left to Bottom-Right
      {
        zero = is_smaller<float>( edges, col, row, col - 1, row - 1 ) ||
               is_smaller<float>( edges, col, row, col + 1, row + 1 );
      }

      out.at<char>( row, col ) = zero ? (char) 0 : (char) 255;
    }
  }
  return out;
}
}
