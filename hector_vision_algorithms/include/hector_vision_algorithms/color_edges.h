//
// Created by Stefan Fabian on 15.06.18.
//

#ifndef HECTOR_VISION_ALGORITHMS_COLOR_EDGES_H
#define HECTOR_VISION_ALGORITHMS_COLOR_EDGES_H

#include <opencv2/core.hpp>

namespace hector_vision_algorithms
{

cv::Mat calculateColorEdges( const cv::Mat &image );

}

#endif //HECTOR_VISION_ALGORITHMS_COLOR_EDGES_H
