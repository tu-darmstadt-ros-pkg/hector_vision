//
// Created by stefan on 09.06.18.
//
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include <chrono>






int main()
{
  cv::Mat input = cv::imread("/home/stefan/argo/src/hector_vision/hector_pipe_detection/start_check/circle4.png");
  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  cv::Mat edges;
  cdm(input, edges);
  std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
  cv::imwrite("/home/stefan/argo/src/hector_vision/hector_pipe_detection/start_check/circle4_cdm.png", edges);
  std::cout << "CDM Took " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0
            << "ms." << std::endl << std::endl;

  start = std::chrono::high_resolution_clock::now();
  color_edges(input, edges);
  end = std::chrono::high_resolution_clock::now();
  cv::imwrite("/home/stefan/argo/src/hector_vision/hector_pipe_detection/start_check/circle4_color_edges.png", edges);
  std::cout << "Color Edges Took " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0
            << "ms." << std::endl;

  return 0;
}