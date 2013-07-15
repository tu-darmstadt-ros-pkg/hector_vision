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

#ifndef HECTOR_QRCODE_DETECTION_H
#define HECTOR_QRCODE_DETECTION_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <tf/transform_listener.h>

namespace zbar {
  class ImageScanner;
}

namespace hector_qrcode_detection {

class qrcode_detection_impl {
public:
  qrcode_detection_impl(ros::NodeHandle nh, ros::NodeHandle priv_nh);
  ~qrcode_detection_impl();

protected:
  void imageCallback(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::CameraInfoConstPtr& camera_info);

private:
  ros::NodeHandle nh_;
  image_transport::ImageTransport image_transport_;
  image_transport::CameraSubscriber camera_subscriber_;
  image_transport::CameraPublisher rotated_image_publisher_;
  image_transport::CameraPublisher qrcode_image_publisher_;

  ros::Publisher percept_publisher_;

  zbar::ImageScanner *scanner_;

  tf::TransformListener *listener_;
  std::string rotation_source_frame_id_;
  std::string rotation_target_frame_id_;
  int rotation_image_size_;
};

} // namespace hector_qrcode_detection

#endif // HECTOR_QRCODE_DETECTION_H
