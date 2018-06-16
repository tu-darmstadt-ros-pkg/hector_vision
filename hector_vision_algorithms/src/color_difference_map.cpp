//
// Created by Stefan Fabian on 15.06.18.
//

#include "hector_vision_algorithms/color_difference_map.h"
#include "hector_vision_algorithms/helpers.h"

#include <opencv2/imgproc.hpp>

namespace hector_vision_algorithms
{

cv::Mat color_difference_map( const cv::Mat &image )
{
  cv::Mat mean_filtered;
  cv::boxFilter(image, mean_filtered, CV_16S, cv::Size(5, 5));

  cv::Mat diagonal_tl_to_br(3, 3, CV_16S, cv::Scalar(0));
  diagonal_tl_to_br.at<short>(0, 0) = 1;
  diagonal_tl_to_br.at<short>(2, 2) = -1;

  cv::Mat vertical_filter(3, 1, CV_16S, cv::Scalar(0));
  vertical_filter.at<short>(0, 0) = 1;
  vertical_filter.at<short>(2, 0) = -1;

  cv::Mat diagonal_tr_to_bl(3, 3, CV_16S, cv::Scalar(0));
  diagonal_tr_to_bl.at<short>(0, 2) = 1;
  diagonal_tr_to_bl.at<short>(2, 0) = -1;

  cv::Mat horizontal_filter(1, 3, CV_16S, cv::Scalar(0));
  horizontal_filter.at<short>(0, 0) = 1;
  horizontal_filter.at<short>(0, 2) = -1;

  cv::Mat max_response;
  applyFilterAndMaxChannel(mean_filtered, max_response, diagonal_tl_to_br);

  cv::Mat response;
  applyFilterAndMaxChannel(mean_filtered, response, vertical_filter);
  max_response = cv::max(max_response, response);

  applyFilterAndMaxChannel(mean_filtered, response, diagonal_tr_to_bl);
  max_response = cv::max(max_response, response);

  applyFilterAndMaxChannel(mean_filtered, response, horizontal_filter);
  max_response = cv::max(max_response, response);

  return max_response;
}
}
