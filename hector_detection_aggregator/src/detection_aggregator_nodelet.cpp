#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <hector_detection_aggregator/detection_aggregator.h>

namespace hector_detection_aggregator {

class DetectionAggregatorNodelet : public nodelet::Nodelet {
  virtual void onInit() {
    ros::NodeHandle &nh = getNodeHandle();
    ros::NodeHandle &pnh = getPrivateNodeHandle();
    detection_aggregator_.reset(new DetectionAggregator(nh, pnh));
    timer_ = nh.createTimer(ros::Duration(1.0/10.0), &DetectionAggregatorNodelet::timerCb, this, false);
  }

  void timerCb(const ros::TimerEvent&) {
    detection_aggregator_->createImage();
  }

  boost::shared_ptr<DetectionAggregator> detection_aggregator_;
  ros::Timer timer_;
};
}

PLUGINLIB_DECLARE_CLASS(hector_detection_aggregator, DetectionAggregatorNodelet,
                        hector_detection_aggregator::DetectionAggregatorNodelet,
                        nodelet::Nodelet);
