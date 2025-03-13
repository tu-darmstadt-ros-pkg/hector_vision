//=================================================================================================
// Copyright (c) 2012, Johannes Meyer, TU Darmstadt
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the Flight Systems and Automatic Control group,
//       TU Darmstadt, nor the names of its contributors may be used to
//       endorse or promote products derived from this software without
//       specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//=================================================================================================

#ifndef HECTOR_TAG_DETECTION_H
#define HECTOR_TAG_DETECTION_H

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

typedef vision_msgs::msg::Detection2DArray Detection2DArray;

namespace zbar {
  class ImageScanner;
}

namespace hector_tag_detection {

class TagDetectionImpl {
public:
  explicit TagDetectionImpl(const rclcpp::Node::SharedPtr& node);
  ~TagDetectionImpl() = default;

protected:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& image, const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info);
  
  void publishEnableStatus() const;
  void enabledCallback(const std_msgs::msg::Bool::ConstSharedPtr& enabled);
  
private:
  rclcpp::Node::SharedPtr node_;

  image_transport::ImageTransport image_transport_;
  zbar::ImageScanner *scanner_;

  rclcpp::TimerBase::SharedPtr check_subscribers_timer_;

  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enabled_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr enabled_pub_;

  image_transport::CameraSubscriber camera_subscriber_;

  image_transport::CameraPublisher tag_image_publisher_;
  rclcpp::Publisher<Detection2DArray>::SharedPtr aggregator_percept_publisher_;

  bool enabled_;
  bool has_subscribers_;

  void startSubscribers();
  void stopSubscribers();
  void publisherSubscriptionCallback();
};

} // namespace hector_tag_detection

#endif // HECTOR_TAG_DETECTION_H
