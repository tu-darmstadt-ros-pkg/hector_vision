//
// Created by Stefan Fabian on 15.06.18.
//

#include "hector_vision_algorithms/thesholding.h"

namespace hector_vision_algorithms
{

template<typename T>
void internalCalculateThresholds( const cv::Mat &mat, double &upper, double &lower, double stop_val ) // TODO special cases can be faster
{
  double old_upper = 0;
  double old_lower = 0;
  double max = 0;
  for ( int r = 0; r < mat.rows; ++r )
  {
    for ( int c = 0; c < mat.cols; ++c )
    {
      if ( mat.at<T>( r, c ) > max )
      {
        max = mat.at<T>( r, c );
      }
    }
  }
  upper = max * 0.5;
  lower = upper / 4;
  while ( cv::abs( upper - old_upper ) > stop_val || cv::abs( lower - old_lower ) > stop_val )
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
}

void calculateThresholds( const cv::Mat &image, double &upper, double &lower )
{
  calculateThresholds(image, upper, lower, 0.1);
}

void calculateThresholds( const cv::Mat &image, double &upper, double &lower, double stop_val )
{
  if ( image.channels() != 1 )
  {
    CV_Error( CV_BadNumChannels, "Calculating an upper and lower threshold only supports single channel images!" );
  }
  switch ( image.type())
  {
    case CV_8S:
      return internalCalculateThresholds<char>( image, upper, lower, stop_val );
    case CV_8U:
      return internalCalculateThresholds<unsigned char>( image, upper, lower, stop_val );
    case CV_16S:
      return internalCalculateThresholds<short>( image, upper, lower, stop_val );
    case CV_16U:
      return internalCalculateThresholds<unsigned short>( image, upper, lower, stop_val );
    case CV_32S:
      return internalCalculateThresholds<int>( image, upper, lower, stop_val );
    case CV_32F:
      return internalCalculateThresholds<float>( image, upper, lower, stop_val );
    case CV_64F:
      return internalCalculateThresholds<double>( image, upper, lower, stop_val );
  }
  CV_Error( CV_BadDepth, "Unknown data type!" );
}

template<typename T>
inline void flowThreshold( const cv::Mat &edges, cv::Mat &binary, double lower, int row, int col )
{
  for ( int i = -1; i <= 1; ++i )
  {
    for ( int j = -1; j <= 1; ++j )
    {
      if ( i == 0 && j == 0 ) continue;
      int r = row + i;
      int c = col + j;
      if ( r < 0 || r == edges.rows ) continue;
      if ( c < 0 || c == edges.cols ) continue;
      if ( binary.at<unsigned char>( r, c ) == 255 || edges.at<T>( r, c ) < lower ) continue;
      binary.at<unsigned char>( r, c ) = 255;
      flowThreshold<T>( edges, binary, lower, r, c );
    }
  }
}

template<typename T>
cv::Mat internalThreshold( const cv::Mat &edges, double upper, double lower )
{
  cv::Mat binary( edges.rows, edges.cols, CV_8U, cv::Scalar( 0 ));
  for ( int row = 0; row < edges.rows; ++row )
  {
    for ( int col = 0; col < edges.cols; ++col )
    {
      if ( edges.at<T>( row, col ) >= upper && binary.at<unsigned char>( row, col ) != 255 )
      {
        binary.at<unsigned char>( row, col ) = 255;
        flowThreshold<T>( edges, binary, lower, row, col );
      }
    }
  }
  return binary;
}

template<typename T>
cv::Mat internalThreshold( const cv::Mat &image, double threshold )
{
  cv::Mat binary( image.dims, image.size.p, CV_8U );
  for ( size_t i = 0; i < image.total(); ++i )
  {
    binary.at<unsigned char>( i ) = image.at<T>( i ) >= threshold ? (char) 255 : (char) 0;
  }
  return binary;
}

cv::Mat threshold( const cv::Mat &image, double threshold )
{
  switch ( image.type())
  {
    case CV_8S:
    case CV_8SC2:
    case CV_8SC3:
    case CV_8SC4:
      return internalThreshold<char>( image, threshold );
    case CV_8U:
    case CV_8UC2:
    case CV_8UC3:
    case CV_8UC4:
      return internalThreshold<unsigned char>( image, threshold );
    case CV_16S:
    case CV_16SC2:
    case CV_16SC3:
    case CV_16SC4:
      return internalThreshold<short>( image, threshold );
    case CV_16U:
    case CV_16UC2:
    case CV_16UC3:
    case CV_16UC4:
      return internalThreshold<unsigned short>( image, threshold );
    case CV_32S:
    case CV_32SC2:
    case CV_32SC3:
    case CV_32SC4:
      return internalThreshold<int>( image, threshold );
    case CV_32F:
    case CV_32FC2:
    case CV_32FC3:
    case CV_32FC4:
      return internalThreshold<float>( image, threshold );
    case CV_64F:
    case CV_64FC2:
    case CV_64FC3:
    case CV_64FC4:
      return internalThreshold<double>( image, threshold );
  }
  CV_Error( CV_BadDepth, "Unknown data type!" );
}

cv::Mat threshold( const cv::Mat &image, double upper, double lower )
{
  if ( image.channels() != 1 )
  {
    CV_Error( CV_BadNumChannels, "Threshold with upper and lower threshold only supports single channel images!" );
  }
  switch ( image.type())
  {
    case CV_8S:
    case CV_8SC2:
    case CV_8SC3:
    case CV_8SC4:
      return internalThreshold<char>( image, upper, lower );
    case CV_8U:
    case CV_8UC2:
    case CV_8UC3:
    case CV_8UC4:
      return internalThreshold<unsigned char>( image, upper, lower );
    case CV_16S:
    case CV_16SC2:
    case CV_16SC3:
    case CV_16SC4:
      return internalThreshold<short>( image, upper, lower );
    case CV_16U:
    case CV_16UC2:
    case CV_16UC3:
    case CV_16UC4:
      return internalThreshold<unsigned short>( image, upper, lower );
    case CV_32S:
    case CV_32SC2:
    case CV_32SC3:
    case CV_32SC4:
      return internalThreshold<int>( image, upper, lower );
    case CV_32F:
    case CV_32FC2:
    case CV_32FC3:
    case CV_32FC4:
      return internalThreshold<float>( image, upper, lower );
    case CV_64F:
    case CV_64FC2:
    case CV_64FC3:
    case CV_64FC4:
      return internalThreshold<double>( image, upper, lower );
  }
  CV_Error( CV_BadDepth, "Unknown data type!" );
}
}
