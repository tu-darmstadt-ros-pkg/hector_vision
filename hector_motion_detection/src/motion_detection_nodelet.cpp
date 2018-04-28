#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <hector_motion_detection/motion_detection.h>

namespace hector_motion_detection {

class MotionDetectionNodelet : public nodelet::Nodelet {
  virtual void onInit() {
    ros::NodeHandle &nh = getNodeHandle();
    ros::NodeHandle &pnh = getPrivateNodeHandle();
    motion_detection_.reset(new MotionDetection(nh, pnh));
  }

  boost::shared_ptr<MotionDetection> motion_detection_;
};
}

PLUGINLIB_EXPORT_CLASS(hector_motion_detection::MotionDetectionNodelet, nodelet::Nodelet)
