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

template<typename FilterType>
void internalCalculateColorEdges( const cv::Mat &image, cv::Mat &edges, cv::Mat &orientation, bool include_orientation,
                                  int filter_type )
{

  cv::Mat x_filter( 3, 3, filter_type, cv::Scalar( 0.f ));
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

  cv::Mat response_y;
  cv::filter2D( image, response_y, filter_type, y_filter );

  cv::Mat jx, jy, jxy;
  getJacobians( response_x, response_y, jx, jy, jxy );

  // compute first (greatest) eigenvalue of 2x2 matrix J'*J.
  // note that the abs() is only needed because some values may be slightly
  // negative due to round-off error.
  cv::Mat d;
  cv::sqrt( cv::abs( jx.mul( jx ) - 2 * jx.mul( jy ) + jy.mul( jy ) + 4 * jxy.mul( jxy )), d );
  cv::Mat e1 = (jx + jy + d) / 2;
  // the 2nd eigenvalue would be:  e2 = (Jx + Jy - D) / 2
  cv::sqrt( e1, edges );

  if ( !include_orientation ) return;

  orientation = cv::Mat( edges.rows, edges.cols, CV_32F );
  for ( int row = 0; row < edges.rows; ++row )
  {
    for ( int col = 0; col < edges.cols; ++col )
    {
      orientation.at<float>( row, col ) = cv::fastAtan2( -jxy.at<float>( row, col ),
                                                         e1.at<float>( row, col ) - jy.at<float>( row, col ));
    }
  }
}

void calculateColorEdges( const cv::Mat &image, cv::Mat &edges, cv::Mat &orientation, bool include_orientation )
{
  if ( image.channels() != 3 )
  {
    char buffer[256];
    sprintf(buffer, "Unsupported number of channels. Only 3 channel color images are supported! "
                    "Number of channels was: %d", image.channels());
    throw std::runtime_error(buffer);
  }
  if ( image.depth() == CV_8U )
  {
    internalCalculateColorEdges<float>( image, edges, orientation, include_orientation, CV_32F );
  }
  else if ( image.depth() == CV_32F )
  {
    internalCalculateColorEdges<float>( image, edges, orientation, include_orientation, CV_32F );
  }
  else
  {
    char buffer[256];
    sprintf(buffer, "Depth not supported! Supported depths are CV_8U and CV_32F. "
                    "Depth was: %d", image.depth());
    throw std::runtime_error( buffer );
  }
}

void calculateColorEdges( const cv::Mat &image, cv::Mat &edges, cv::Mat &orientation )
{
  calculateColorEdges( image, edges, orientation, true );
}

void calculateColorEdges( const cv::Mat &image, cv::Mat &edges )
{
  cv::Mat orientation;
  calculateColorEdges( image, edges, orientation, false );
}
}
