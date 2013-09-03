//=================================================================================================
// Copyright (c) 2011, Stefan Kohlbrecher, TU Darmstadt
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

#include <ros/ros.h>

#include <hector_thermal_self_filter/hector_thermal_self_filter.h>
#include <hector_worldmodel_msgs/VerifyPercept.h>

namespace hector_thermal_self_filter{

class ThermalSelfFilter
{
public:
  ThermalSelfFilter()
  {
    ros::NodeHandle pnh("~");
    self_filter_ = new HectorThermalSelfFilter(pnh, &tfL_);

    ooi_verification_service_ = pnh.advertiseService("verify_percept", &ThermalSelfFilter::verifyPerceptCallBack, this);
  }

  ~ThermalSelfFilter()
  {
    delete self_filter_;
  }

  bool verifyPerceptCallBack(hector_worldmodel_msgs::VerifyPercept::Request  &req,
                            hector_worldmodel_msgs::VerifyPercept::Response &res )
  {
    bool belongsToRobot = self_filter_->pointBelongsToRobot(req.percept.pose.pose.position, req.percept.header);

    if (belongsToRobot){
      res.response = hector_worldmodel_msgs::VerifyPerceptResponse::DISCARD;
    }else{
      res.response = hector_worldmodel_msgs::VerifyPerceptResponse::UNKNOWN;
    }

    return true;
  }

protected:
  HectorThermalSelfFilter* self_filter_;
  tf::TransformListener tfL_;

  ros::ServiceServer ooi_verification_service_;

};

}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "thermal_self_filter");

  hector_thermal_self_filter::ThermalSelfFilter sf;

  ros::spin();

  return 0;
}
