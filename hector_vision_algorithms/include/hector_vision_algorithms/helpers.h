//
// Created by Stefan Fabian on 15.06.18.
//

#ifndef HECTOR_VISION_ALGORITHMS_HELPERS_H
#define HECTOR_VISION_ALGORITHMS_HELPERS_H

#include <opencv2/core.hpp>

namespace hector_vision_algorithms
{

void applyFilterAndMaxChannel( const cv::Mat &mat, cv::Mat &out, const cv::Mat &filter );
}

#endif //HECTOR_VISION_ALGORITHMS_HELPERS_H
