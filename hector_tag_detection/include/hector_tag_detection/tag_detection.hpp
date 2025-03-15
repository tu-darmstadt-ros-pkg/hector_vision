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

#include <zbar.h>
#include <apriltag/apriltag.h>
#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

typedef vision_msgs::msg::Detection2DArray Detection2DArray;

namespace hector_tag_detection {

class TagDetectionImpl {
public:
  explicit TagDetectionImpl(const rclcpp::Node::SharedPtr& node);
  TagDetectionImpl(TagDetectionImpl& td) = delete;
  TagDetectionImpl(TagDetectionImpl&& td) = delete;
  ~TagDetectionImpl() = default;

protected:
  /**
   * Called when an image is received.
   * Performs tag detection on received image and publishes detected tags.
   * @param image The tag detection is performed on this image
   * @param camera_info The camera info is replicated for the debug crops but is otherwise unused
   */
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& image, const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info);

  /**
   * Publishes whether the node is enabled. Called once on startup and by enabledCallback
   */
  void publishEnableStatus() const;
  /**
   * Enables or disables the node, stopping or starting subscribers.
   * Called either by parameter change or topic callback via msgEnabledCallback.
   * @param enabled Enables or disables the node
   */
  void enabledCallback(const bool& enabled);
  /**
   * Enables or disables the node via enabledCallback
   * @param enabled Enables or disables the node
   */
  void msgEnabledCallback(const std_msgs::msg::Bool::ConstSharedPtr& enabled);
  
private:
  bool enabled_;
  bool has_subscribers_;

  zbar::ImageScanner *qrcode_detector_;
  std::shared_ptr<apriltag_detector_t> apriltag_detector_;
  std::shared_ptr<apriltag_family_t> apriltag_family_;

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<rclcpp::ParameterEventHandler> parameter_event_handler_;
  rclcpp::ParameterCallbackHandle::SharedPtr enabled_callback_handle_;
  image_transport::ImageTransport image_transport_;
  std::shared_ptr<rclcpp::ParameterEventHandler> param_subscriber_;
  rclcpp::TimerBase::SharedPtr check_subscribers_timer_;

  image_transport::CameraSubscriber camera_subscriber_;

  image_transport::CameraPublisher tag_image_publisher_;
  rclcpp::Publisher<Detection2DArray>::SharedPtr aggregator_percept_publisher_;

  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enabled_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr enabled_pub_;
  
  /**
   * Starts the node's subscriptions when the node is enabled and has subscribero of its own.
   */
  void startSubscribers();
  /**
   * Stops the node's subscribers when the node is disabled or has no subscribers of its own.
   */
  void stopSubscribers();
  /**
   * Called periodically to check if the node's publishers have any subscribers,
   * starting or stopping the node's own subscription accordingly.
   */
  void publisherSubscriptionCallback();
};

} // namespace hector_tag_detection

#endif // HECTOR_TAG_DETECTION_H
