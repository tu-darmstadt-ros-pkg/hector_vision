//
// Created by Stefan Fabian on 15.06.18.
//

#ifndef HECTOR_VISION_ALGORITHMS_THESHOLDING_H
#define HECTOR_VISION_ALGORITHMS_THESHOLDING_H

#include <opencv2/core.hpp>

namespace hector_vision_algorithms
{

void calculateThresholds( const cv::Mat &image, double &upper, double &lower );

void calculateThresholds( const cv::Mat &image, double &upper, double &lower, double stop_val );

cv::Mat threshold( const cv::Mat &image, double threshold );

cv::Mat threshold( const cv::Mat &image, double upper, double lower );
}

#endif //HECTOR_VISION_ALGORITHMS_THESHOLDING_H
