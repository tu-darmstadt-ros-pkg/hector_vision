//
// Created by Stefan Fabian on 15.06.18.
//

#include "hector_vision_algorithms/helpers.h"

#include <opencv2/imgproc.hpp>

namespace hector_vision_algorithms
{

void applyFilterAndMaxChannel( const cv::Mat &mat, cv::Mat &out, const cv::Mat &filter )
{
  cv::Mat channels[3];
  cv::Mat response;
  cv::filter2D(mat, response, -1, filter);
  cv::split(cv::abs(response), channels);
  out = cv::max(channels[0], cv::max(channels[1], channels[2]));
}
}
