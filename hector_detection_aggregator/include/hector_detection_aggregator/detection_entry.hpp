//
// Created by finn on 09.12.25.
//

#ifndef DETECTION_ENTRY_HPP
#define DETECTION_ENTRY_HPP

#include <string>
#include <vision_msgs/msg/detection2_d_array.hpp>

namespace hector_detection_aggregator
{

struct DetectionEntry {
  std::string type;
  std::string id;
  vision_msgs::msg::Detection2D detection;

  DetectionEntry( std::string type, std::string id, vision_msgs::msg::Detection2D detection )
      : type( std::move( type ) ), id( std::move( id ) ), detection( std::move( detection ) )
  {
  }
};

class DetectionEntry2
{
  DetectionEntry2( vision_msgs::msg::Detection2D detection );

  std::string type;
  std::string id;
};

} // namespace hector_detection_aggregator

#endif // DETECTION_ENTRY_HPP
