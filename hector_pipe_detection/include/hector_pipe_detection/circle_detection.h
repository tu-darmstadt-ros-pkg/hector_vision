//
// Created by stefan on 11.06.18.
//

#ifndef HECTOR_PIPE_DETECTION_CIRCLE_DETECTION_H
#define HECTOR_PIPE_DETECTION_CIRCLE_DETECTION_H

#include <opencv2/core/core.hpp>
#include <memory>
#include <image_transport/publisher.h>
#include <image_transport/image_transport.h>

namespace hector_pipe_detection
{

class DebugInfo
{
public:
  explicit DebugInfo( std::shared_ptr<image_transport::ImageTransport> transport );

  void publishEdgeImage( const cv::Mat &edge_image );

  void publishFilteredContours( const cv::Mat &image, std::vector<std::vector<cv::Point>> contours );

  void publishSubEdgeImage( const cv::Mat &sub_edge_image );

  void publishSubImageContours( const cv::Mat &sub_image, std::vector<std::vector<cv::Point>> contours,
                                const cv::Point &offset );

  void publishDetection( const cv::Mat &image, const cv::Point2d &center, double radius );

private:
  image_transport::Publisher edge_image_pub_;
  image_transport::Publisher filtered_contours_pub_;
  image_transport::Publisher sub_edge_image_pub_;
  image_transport::Publisher sub_image_contours_pub_;
  image_transport::Publisher detection_pub_;
};

void cdm( const cv::Mat &image, cv::Mat &out );

bool findOuterCircle( const cv::Mat &image, int downsample_passes, cv::Point2d &center, double &radius,
                      std::shared_ptr<DebugInfo> debug_info = nullptr );
}

#endif //HECTOR_PIPE_DETECTION_CIRCLE_DETECTION_H
