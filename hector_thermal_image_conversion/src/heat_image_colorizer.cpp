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

#include <hector_thermal_image_conversion/heat_image_colorizer.h>

HeatImageColorizer::HeatImageColorizer(ros::NodeHandle& nh_,ros::NodeHandle& pnh_)
{
  // This converter does not require camera info, so we just subscribe to the image
  // Setup is inspired by standard image_proc nodelets such as
  // https://github.com/strawlab/image_pipeline/blob/master/image_proc/src/nodelets/debayer.cpp
  it_.reset(new image_transport::ImageTransport(pnh_));

  typedef image_transport::SubscriberStatusCallback ConnectCB;
  ConnectCB connect_cb = boost::bind(&HeatImageColorizer::connectCb, this);

  boost::lock_guard<boost::mutex> lock(connect_mutex_);
  converted_image_pub_ = it_->advertise("image_mapped", 1, connect_cb, connect_cb);

  color_mapping_ = cv::Mat(iron_bow_color_mapping, true);
}

void HeatImageColorizer::connectCb()
{
  boost::lock_guard<boost::mutex> lock(connect_mutex_);
  if (converted_image_pub_.getNumSubscribers() == 0)
    image_sub_.shutdown();
  else if (!image_sub_)
    image_sub_ = it_->subscribe("image", 1, &HeatImageColorizer::imageCb, this);
}

void HeatImageColorizer::colorizeImage(const sensor_msgs::ImageConstPtr& image_msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO16);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cv::Mat colorized_image;

  cv::normalize(cv_ptr->image, colorized_image, 0, 255, cv::NORM_MINMAX, CV_8U);
  cvtColor(colorized_image, colorized_image, cv::COLOR_GRAY2RGB);
  cv::LUT(colorized_image, iron_bow_color_mapping, colorized_image);

  cv_ptr->image = colorized_image;
  cv_ptr->encoding = "rgb8";
  converted_image_pub_.publish(cv_ptr->toImageMsg());
}

void HeatImageColorizer::imageCb(const sensor_msgs::ImageConstPtr& image_msg)
{
  colorizeImage(image_msg);
}
