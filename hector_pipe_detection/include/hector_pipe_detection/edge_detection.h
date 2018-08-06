//
// Created by stefan on 12.06.18.
//

#ifndef HECTOR_PIPE_DETECTION_EDGE_DETECTION_H
#define HECTOR_PIPE_DETECTION_EDGE_DETECTION_H

#include <opencv2/core/core.hpp>

namespace hector_pipe_detection
{


void color_edges(const cv::Mat &image, cv::Mat &edges);

}

#endif //HECTOR_PIPE_DETECTION_EDGE_DETECTION_H
