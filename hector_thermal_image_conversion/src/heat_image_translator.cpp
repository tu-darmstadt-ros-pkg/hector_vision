//=================================================================================================
// Copyright (c) 2015, Stefan Kohlbrecher, TU Darmstadt
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

#include <hector_thermal_image_conversion/heat_image_translator.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <boost/thread.hpp>
#include <nodelet/nodelet.h>

HeatImageTranslator::HeatImageTranslator(ros::NodeHandle& nh_,ros::NodeHandle& pnh_)
{
  pnh_.param<bool>("use_raw_threshold", use_raw_threshold_, true);

  if (use_raw_threshold_)
  {
    pnh_.param<double>("min_temp_img", min_temp_img_, 22000.0);
    pnh_.param<double>("max_temp_img", max_temp_img_, 25000.0);
  }
  else
  {
    pnh_.param<double>("min_temp_img", min_temp_img_, 10.0);
    pnh_.param<double>("max_temp_img", max_temp_img_, 200.0);
  }

  pnh_.param<double>("temperature_unit_kelvin", temperature_unit_kelvin_, 0.04);

  mappingDefined_ = true;

  // This converter does not require camera info, so we just subscribe to the image
  // Setup is inspired by standard image_proc nodelets such as
  // https://github.com/strawlab/image_pipeline/blob/master/image_proc/src/nodelets/debayer.cpp
  it_.reset(new image_transport::ImageTransport(pnh_));

  typedef image_transport::SubscriberStatusCallback ConnectCB;
  ConnectCB connect_cb = boost::bind(&HeatImageTranslator::connectCb, this);

  boost::lock_guard<boost::mutex> lock(connect_mutex_);
  converted_image_pub_ = it_->advertise("image_converted", 1, connect_cb, connect_cb);

  dyn_rec_callback_type_ = boost::bind(&HeatImageTranslator::dynRecCallback, this, _1, _2);
  dyn_rec_server_.setCallback(dyn_rec_callback_type_);
}

void HeatImageTranslator::connectCb()
{
  boost::lock_guard<boost::mutex> lock(connect_mutex_);
  if (converted_image_pub_.getNumSubscribers() == 0)
    image_sub_.shutdown();
  else if (!image_sub_)
    image_sub_ = it_->subscribe("image", 1, &HeatImageTranslator::imageCb, this);
}

void HeatImageTranslator::convertImage(const sensor_msgs::ImageConstPtr& image_msg)
{
  //ROS_ERROR("cb");
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO16);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat convertedImage;
  int raw_min = 0;
  int raw_max = 0;

  if (use_raw_threshold_)
  {
    raw_min = min_temp_img_;
    raw_max = max_temp_img_;
  }
  else
  {
    raw_min = (min_temp_img_+273.15)/temperature_unit_kelvin_;
    raw_max = (max_temp_img_+273.15)/temperature_unit_kelvin_;
  }

  const double alpha = 255.0 / (raw_max - raw_min);
  const double beta = -alpha * raw_min;
  cv_ptr->image.convertTo(convertedImage, CV_8UC1, alpha, beta);

  cv_ptr->image = convertedImage;
  cv_ptr->encoding = "mono8";
  converted_image_pub_.publish(cv_ptr->toImageMsg());
}

void HeatImageTranslator::imageCb(const sensor_msgs::ImageConstPtr& image_msg)
{
  convertImage(image_msg);
}

void HeatImageTranslator::dynRecCallback(DynRecConfig &config, uint32_t level)
{
  min_temp_img_ = config.min_value;
  max_temp_img_ = config.max_value;
  use_raw_threshold_ = config.use_raw_threshold;
}

