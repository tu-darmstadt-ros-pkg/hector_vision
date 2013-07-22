//=================================================================================================
// Copyright (c) 2012, Stefan Kohlbrecher, TU Darmstadt
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

#ifndef __HectorThermalSelfFilter_h_
#define __HectorThermalSelfFilter_h_

#include <pcl_conversions/pcl_conversions.h>
#include <tf/transform_listener.h>
#include <robot_self_filter/self_see_filter.h>

class HectorThermalSelfFilter{
public:

  HectorThermalSelfFilter(ros::NodeHandle& nh, tf::TransformListener* tfl_in)
    : tfL_(tfl_in)
  {
    self_filter_ = new filters::SelfFilter<pcl::PointCloud<pcl::PointXYZ> > (nh);


    self_filter_->getSelfMask()->getLinkNames(frames_);
    if (frames_.empty()){
      ROS_ERROR ("No valid frames have been passed into the self filter.");
    }else{
      ROS_INFO ("Self filter uses the following links to filter:");

      for (size_t i = 0; i < frames_.size(); ++i){
        ROS_INFO("Filtered frame %u : %s", static_cast<unsigned int>(i), frames_[i].c_str());
      }
    }
  }

  virtual ~HectorThermalSelfFilter(){
    delete self_filter_;
  }

  bool pointBelongsToRobot(const geometry_msgs::Point& point_in, const std_msgs::Header& header)//Define argument types)
  {
    pcl::PointCloud<pcl::PointXYZ> cloud_in;
    pcl_conversions::toPCL(header, cloud_in.header);
    cloud_in.push_back(pcl::PointXYZ(point_in.x, point_in.y, point_in.z));

    pcl::PointCloud<pcl::PointXYZ> cloud_filtered;

    if (waitForRelevantTransforms(header)){
      self_filter_->updateWithSensorFrame (cloud_in, cloud_filtered, header.frame_id);

      if(cloud_filtered.size() > 0){
        return false;
      }else{
        return true;
      }
    }else{
      return false;
    }
  }

  // Returns false if waiting failed with a tf exception
  bool waitForRelevantTransforms(const std_msgs::Header& header)
  {
    try{
      size_t num_frames = frames_.size();
      ros::Duration waitDuration(0.5);

      for (size_t i = 0; i < num_frames; ++i){
        tfL_->waitForTransform(header.frame_id, frames_[i], header.stamp, waitDuration);
      }
    } catch (tf::TransformException ex) {
      ROS_ERROR("Self filter failed waiting for necessary transforms: %s", ex.what());
      return false;
    }
    return true;
  }





protected:
  filters::SelfFilter<pcl::PointCloud<pcl::PointXYZ> > *self_filter_;
  std::vector<std::string> frames_;

  tf::TransformListener* tfL_;


};

#endif
