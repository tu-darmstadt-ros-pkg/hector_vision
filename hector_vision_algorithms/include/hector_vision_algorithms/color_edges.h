//
// Created by Stefan Fabian on 15.06.18.
//

#ifndef HECTOR_VISION_ALGORITHMS_COLOR_EDGES_H
#define HECTOR_VISION_ALGORITHMS_COLOR_EDGES_H

#include <opencv2/core.hpp>

namespace hector_vision_algorithms
{

/*!
 * Uses the largest eigenvalue of the Jacobian to determine the edge strength.
 * Taken from:
 * http://www.mathworks.com/matlabcentral/fileexchange/28114-fast-edges-of-a-color-image-actual-color-not-converting-to-grayscale/content/coloredges.m
 * @param image The input image. Usually CV_8UC3 (RGB8) but may also be CV_32FC3.
 * @param edges The output edge image. Format: CV_32FC3.
 * @param orientation The orientation of the edges. Format: CV_32F. Values: 0 to 360 where 0 is the positive x-axis and values increase counter-clockwise.
 */
void calculateColorEdges( const cv::Mat &image, cv::Mat &edges, cv::Mat &orientation );

/*!
 * See calculateColorEdges( const cv::Mat &, cv::Mat &, cv::Mat & ).
 * @param image The input image. Usually CV_8UC3 (RGB8) but may also be CV_32FC3.
 * @param edges The output edge image. Format: CV_32FC3.
 */
void calculateColorEdges( const cv::Mat &image, cv::Mat &edges );
}

#endif //HECTOR_VISION_ALGORITHMS_COLOR_EDGES_H
