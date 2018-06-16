//
// Created by Stefan Fabian on 12.06.18.
//

#include "hector_pipe_detection/edge_detection.h"

#include <opencv2/opencv.hpp>

namespace hector_pipe_detection
{

template<typename T>
void calculateThresholds( const cv::Mat &mat, double &upper, double &lower )
{
  double old_upper = 0;
  double old_lower = 0;
  upper = 160;
  lower = 80;
  while ( cv::abs( upper - old_upper ) > 0.1 || cv::abs( lower - old_lower ) > 0.1 )
  {
    old_upper = upper;
    old_lower = lower;
    int count_upper = 0;
    int count_middle = 0;
    int count_lower = 0;
    double upper_sum = 0;
    double middle_sum = 0;
    double lower_sum = 0;
    for ( int row = 0; row < mat.rows; ++row )
    {
      for ( int col = 0; col < mat.cols; ++col )
      {
        T val = mat.at<T>( row, col );
        if ( val > upper )
        {
          ++count_upper;
          upper_sum += val;
        }
        else if ( val > lower )
        {
          ++count_middle;
          middle_sum += val;
        }
        else
        {
          ++count_lower;
          lower_sum += val;
        }
      }
    }
    upper = (upper_sum / count_upper + middle_sum / count_middle) / 2;
    lower = (middle_sum / count_middle + lower_sum / count_lower) / 2;
  }
//  std::cout << "Lower threshold: " << lower << ", Upper: " << upper << std::endl;
}

void applyFilterAndMaxChannel( const cv::Mat &mat, cv::Mat &out, const cv::Mat &filter )
{
  cv::Mat channels[3];
  cv::Mat response;
  cv::filter2D( mat, response, -1, filter );
  cv::split( cv::abs( response ), channels );
  out = cv::max( channels[0], cv::max( channels[1], channels[2] ));
}

void combineResponses( cv::Mat &out, const cv::Mat &in, cv::Mat &orientation, const char orientation_in )
{
  for ( int row = 0; row < out.rows; ++row )
  {
    for ( int col = 0; col < out.cols; ++col )
    {
      if ( out.at<short>( row, col ) > in.at<short>( row, col )) continue;
      out.at<short>( row, col ) = in.at<short>( row, col );
      orientation.at<char>( row, col ) = orientation_in;
    }
  }
}

template<typename T>
inline bool isSmaller( const cv::Mat &mat, int x1, int y1, int x2, int y2 )
{
  if ( x2 < 0 || x2 >= mat.cols ) return false;
  if ( y2 < 0 || y2 >= mat.rows ) return false;
  return mat.at<T>( y1, x1 ) <= mat.at<T>( y2, x2 );
}

void cdm( const cv::Mat &image, cv::Mat &out )
{
  cv::Mat mean_filtered;
  cv::boxFilter( image, mean_filtered, CV_16S, cv::Size( 5, 5 ));

  cv::Mat diagonal_tl_to_br( 3, 3, CV_16S, cv::Scalar( 0 ));
  diagonal_tl_to_br.at<short>( 0, 0 ) = 1;
  diagonal_tl_to_br.at<short>( 2, 2 ) = -1;

  cv::Mat vertical_filter( 3, 1, CV_16S, cv::Scalar( 0 ));
  vertical_filter.at<short>( 0, 0 ) = 1;
  vertical_filter.at<short>( 2, 0 ) = -1;

  cv::Mat diagonal_tr_to_bl( 3, 3, CV_16S, cv::Scalar( 0 ));
  diagonal_tr_to_bl.at<short>( 0, 2 ) = 1;
  diagonal_tr_to_bl.at<short>( 2, 0 ) = -1;

  cv::Mat horizontal_filter( 1, 3, CV_16S, cv::Scalar( 0 ));
  horizontal_filter.at<short>( 0, 0 ) = 1;
  horizontal_filter.at<short>( 0, 2 ) = -1;

  cv::Mat max_response;
  applyFilterAndMaxChannel( mean_filtered, max_response, diagonal_tl_to_br );
  // Orientations: 0: TL2BR, 1: Vertical, 2: BL2TR, 3: Horizontal
  cv::Mat orientation( max_response.rows, max_response.cols, CV_8S, cv::Scalar( 0 ));

  cv::Mat response;
  applyFilterAndMaxChannel( mean_filtered, response, vertical_filter );
  combineResponses( max_response, response, orientation, 1 );

  applyFilterAndMaxChannel( mean_filtered, response, diagonal_tr_to_bl );
  combineResponses( max_response, response, orientation, 2 );

  applyFilterAndMaxChannel( mean_filtered, response, horizontal_filter );
  combineResponses( max_response, response, orientation, 3 );

  cv::Mat mean, stddev;
  cv::meanStdDev( max_response, mean, stddev );

  out = cv::Mat( max_response.rows, max_response.cols, CV_8U, cv::Scalar( 0 ));
  for ( int row = 0; row < max_response.rows; ++row )
  {
    for ( int col = 0; col < max_response.cols; ++col )
    {
      if ( max_response.at<short>( row, col ) < mean.at<double>( 0, 0 ))
      {
        continue;
      }
      bool zero;
      switch ( orientation.at<char>( row, col ))
      {
        case 0:
          zero = isSmaller<short>( max_response, col, row, col - 1, row + 1 ) ||
                 isSmaller<short>( max_response, col, row, col + 1, row - 1 );
          break;
        case 1:
          zero = isSmaller<short>( max_response, col, row, col, row + 1 ) ||
                 isSmaller<short>( max_response, col, row, col, row - 1 );
          break;
        case 2:
          zero = isSmaller<short>( max_response, col, row, col - 1, row - 1 ) ||
                 isSmaller<short>( max_response, col, row, col + 1, row + 1 );
          break;
        case 3:
          zero = isSmaller<short>( max_response, col, row, col - 1, row ) ||
                 isSmaller<short>( max_response, col, row, col + 1, row );
          break;
        default:
          zero = true;
      }
      if ( zero ) continue;
      out.at<unsigned char>( row, col ) = 255;
    }
  }

//  cv::threshold( max_response, max_response, mean.at<double>( 0, 0 ) + stddev.at<double>( 0, 0 ), 255,
//                 CV_THRESH_TOZERO );
//  max_response.convertTo( out, CV_8U );
}

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

template<typename T>
inline void flowThreshold(const cv::Mat &edges, cv::Mat &binary, double lower, int row, int col)
{
  for (int i = -1; i <= 1; ++i)
  {
    for (int j = -1; j <= 1; ++j)
    {
      if (i == 0 && j == 0) continue;
      int r = row + i;
      int c = col + j;
      if (r < 0 || r == edges.rows) continue;
      if (c < 0 || c == edges.cols) continue;
      if (binary.at<unsigned char>(r, c) == 255 || edges.at<T>(r, c) < lower) continue;
      binary.at<unsigned char>(r, c) = 255;
      flowThreshold<T>(edges, binary, lower, r, c);
    }
  }
}

template<typename T>
void threshold(const cv::Mat &edges, cv::Mat &binary, double upper, double lower)
{
  binary = cv::Mat(edges.rows, edges.cols, CV_8U, cv::Scalar(0));
  for (int row = 0; row < edges.rows; ++row)
  {
    for (int col = 0; col < edges.cols; ++col)
    {
      if ( edges.at<T>( row, col ) >= upper && binary.at<unsigned char>( row, col ) != 255 )
      {
        binary.at<unsigned char>( row, col ) = 255;
        flowThreshold<T>( edges, binary, lower, row, col );
      }
    }
  }
}

void color_edges( const cv::Mat &image, cv::Mat &out )
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
  double upper;
  double lower;
  calculateThresholds<float>( edges, upper, lower );
  threshold<float>(edges, out, upper, lower);
  //cv::threshold( edges, out, upper, 255, CV_THRESH_BINARY );
  out.convertTo( out, CV_8U );
  return;

  /*cv::Mat mean, stddev;
  cv::meanStdDev( edges, mean, stddev );
  double high_threshold = mean.at<double>( 0, 0 ) + 2 * stddev.at<double>( 0, 0 );
  double low_threshold = mean.at<double>( 0, 0 ) + 0.5 * stddev.at<double>( 0, 0 );
  out = cv::Mat( edges.rows, edges.cols, CV_8U );
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
        if ( edges.at<float>( row, col ) >= low_threshold )
        {
          out.at<char>( row, col ) = 127;
          continue;
        }
        zero = true;
      }
      else if ( orientation > 67.5 || orientation <= -67.5 ) // Vertical
      {
        zero = isSmaller<float>( edges, col, row, col, row - 1 ) || isSmaller<float>( edges, col, row, col, row + 1 );
      }
      else if ( orientation > 22.5 ) // Bottom-Left to Top-Right
      {
        zero = isSmaller<float>( edges, col, row, col + 1, row - 1 ) ||
               isSmaller<float>( edges, col, row, col - 1, row + 1 );
      }
      else if ( orientation > -22.5 ) // Horizontal
      {
        zero = isSmaller<float>( edges, col, row, col - 1, row ) || isSmaller<float>( edges, col, row, col + 1, row );
      }
      else // Top-Left to Bottom-Right
      {
        zero = isSmaller<float>( edges, col, row, col - 1, row - 1 ) ||
               isSmaller<float>( edges, col, row, col + 1, row + 1 );
      }

      out.at<char>( row, col ) = zero ? (char) 0 : (char) 255;
    }
  }*/
}
}
