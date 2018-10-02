//
// Created by Stefan Fabian on 15.06.18.
//
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/tuple.hpp>

#include <hector_vision_algorithms/color_difference_map.h>
#include <hector_vision_algorithms/color_edges.h>
#include <hector_vision_algorithms/thesholding.h>

#include "../np_opencv_converter/np_opencv_converter.hpp"

//class GenericWrapper {
//public:
//  GenericWrapper(const int& _var_int = 1, const float& _var_float = 1.f,
//                 const double& _var_double = 1.d, const std::string& _var_string = std::string("test_string"))
//    : var_int(_var_int), var_float(_var_float), var_double(_var_double), var_string(_var_string)
//  {
//
//  }
//
//  cv::Mat process(const cv::Mat& in) {
//    std::cerr << "in: " << in << std::endl;
//    std::cerr << "sz: " << in.size() << std::endl;
//    std::cerr << "Returning transpose" << std::endl;
//    return in.t();
//  }
//
//private:
//  int var_int;
//  float var_float;
//  double var_double;
//  std::string var_string;
//};

boost::python::tuple calculateColorEdges( const cv::Mat &image, bool include_orientation )
{
  cv::Mat edges, orientation;
  if ( include_orientation )
  {
    hector_vision_algorithms::calculateColorEdges( image, edges, orientation );
  }
  else
  {
    hector_vision_algorithms::calculateColorEdges( image, edges );
  }
  return boost::python::make_tuple( edges, orientation );
}

boost::python::tuple calculateColorEdges( const cv::Mat &image )
{
  return calculateColorEdges( image, true );
}

boost::python::tuple calculateThresholds( const cv::Mat &image )
{
  double upper, lower;
  hector_vision_algorithms::calculateThresholds( image, upper, lower );
  return boost::python::make_tuple( upper, lower );
}

boost::python::tuple calculateThresholds( const cv::Mat &image, double stop_val )
{
  double upper, lower;
  hector_vision_algorithms::calculateThresholds( image, upper, lower, stop_val );
  return boost::python::make_tuple( upper, lower );
}

// The module name here *must* match the name of the python project
// specified in the CMakeLists.txt file with lib pasted on the front:
// as in libPY_PROJECT_NAME
BOOST_PYTHON_MODULE(libhector_vision_algorithms_py)
{
  // This using statement is just for convenience
  using namespace boost::python;

  fs::python::init_and_export_converters();

//  class_<GenericWrapper>( "GenericWrapper" )
//    .def( init<optional<int, float, double, std::string> >(
//      (arg( "var_int" ) = 1, arg( "var_float" ) = 1.f, arg( "var_double" ) = 1.d,
//       arg( "var_string" ) = std::string( "test" ))))
//    .def( "process", &GenericWrapper::process );

  // Export functions and documentation strings
  // Edge Algorithms
  def( "color_difference_map", &hector_vision_algorithms::calculateColorDifferenceMap,
       "edges = color_difference_map( image )" );
  def<tuple ( * )( const cv::Mat & )>( "color_edges", &calculateColorEdges, "edges, orientation = color_edges( image )" );
  def<tuple ( * )( const cv::Mat &, bool )>( "color_edges", &calculateColorEdges, "edges, orientation = color_edges( image, include_orientation )" );

  // Thresholding
  def<tuple ( * )( const cv::Mat & )>( "calculate_thresholds", &calculateThresholds,
                                       "upper, lower = calculate_thresholds( image )" );
  def<tuple ( * )( const cv::Mat &, double )>( "calculate_thresholds", &calculateThresholds,
                                               "upper, lower = calculate_thresholds( image, stop_val )" );
  def<cv::Mat ( * )( const cv::Mat &, double )>( "threshold", &hector_vision_algorithms::threshold,
                                                 "binary_image = threshold( image, thresh )" );
  def<cv::Mat ( * )( const cv::Mat &, double, double )>( "threshold", &hector_vision_algorithms::threshold,
                                                         "binary_image = threshold( image, upper, lower )" );

  // Export classes
  //class_< AwesomeRobot, boost::shared_ptr<AwesomeRobot> >( "AwesomeRobot", init<>() )
  //.def("isAwesome", &AwesomeRobot::isAwesome, "bool robot.isAwesome() returns true if this robot is awesome");
}
