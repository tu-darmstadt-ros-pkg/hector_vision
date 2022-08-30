//=================================================================================================
// Copyright (c) 2019, Stefan Kohlbrecher and Marius Schnaubelt, TU Darmstadt
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the Simulation, Systems Optimization and Robotics
//       group, TU Darmstadt nor the names of its contributors may be used to
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


#ifndef HEAT_IMAGE_COLORIZER_H__
#define HEAT_IMAGE_COLORIZER_H__

#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <image_transport/camera_subscriber.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <boost/thread.hpp>
#include <nodelet/nodelet.h>

#include "iron_bow_color_mapping.h"

class HeatImageColorizer
{
public:
    HeatImageColorizer(ros::NodeHandle& nh_,ros::NodeHandle& pnh_);

  void connectCb();

  void imageCb(const sensor_msgs::ImageConstPtr& image_msg);

  void colorizeImage(const sensor_msgs::ImageConstPtr& image_msg);

protected:

  cv::Mat tmp_grey_img_;
  cv::Mat tmp_rgb_img_;
  cv_bridge::CvImagePtr img_out_;

  boost::mutex connect_mutex_;

  boost::shared_ptr<image_transport::ImageTransport> it_;
  image_transport::Subscriber image_sub_;

  image_transport::Publisher converted_image_pub_;
};

#endif // HEAT_IMAGE_COLORIZER_H__
