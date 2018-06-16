//
// Created by Stefan Fabian on 11.06.18.
//

#include "hector_pipe_detection/circle_detection.h"

#include <hector_profiling/timer.h>

#include <nodelet/nodelet.h>

#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>

#include <image_geometry/pinhole_camera_model.h>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>

#include <atomic>
#include <chrono>

namespace hector_pipe_detection
{


class PipeDetectionNodelet : public nodelet::Nodelet
{
public:
  PipeDetectionNodelet();

protected:

  void onInit() override;

  void detectCallback( const sensor_msgs::ImageConstPtr &image, const sensor_msgs::CameraInfoConstPtr &camera_info );

  void enableCallback();

  void disableCallback();

  double circle_radius_;
  std::shared_ptr<image_transport::ImageTransport> image_transport_;
  std::unique_ptr<image_geometry::PinholeCameraModel> camera_model_;
  image_transport::CameraSubscriber image_subscriber_;
  ros::Publisher pose_publisher_;
  ros::Publisher pose_2d_publisher;

  std::atomic<int> subscriber_count_;

  std::shared_ptr<DebugInfo> debug_info_;
};

PipeDetectionNodelet::PipeDetectionNodelet() : subscriber_count_( 0 )
{
}

void PipeDetectionNodelet::onInit()
{
  ros::NodeHandle &nh = getNodeHandle();
  image_transport_ = std::make_shared<image_transport::ImageTransport>( nh );
  ros::NodeHandle &pnh = getPrivateNodeHandle();
  pose_publisher_ = pnh.advertise<geometry_msgs::PoseStamped>( "pose", 1,
                                                               boost::bind( &PipeDetectionNodelet::enableCallback,
                                                                            this ),
                                                               boost::bind( &PipeDetectionNodelet::disableCallback,
                                                                            this ),
                                                               ros::VoidConstPtr(),
                                                               true );
  pose_2d_publisher = pnh.advertise<geometry_msgs::PoseStamped>( "pose_2d", 1,
                                                                 boost::bind( &PipeDetectionNodelet::enableCallback,
                                                                              this ),
                                                                 boost::bind( &PipeDetectionNodelet::disableCallback,
                                                                              this ),
                                                                 ros::VoidConstPtr(),
                                                                 true );

  // TODO: Make param for this
  if (pnh.param("debug", false))
  {
    debug_info_ = std::make_shared<DebugInfo>( std::make_shared<image_transport::ImageTransport>( pnh ));
  }

  circle_radius_ = pnh.param("circle_radius", 0.025);
  NODELET_INFO( "PipeDetection initialized." );
}

void PipeDetectionNodelet::detectCallback( const sensor_msgs::ImageConstPtr &msg,
                                           const sensor_msgs::CameraInfoConstPtr &camera_info )
{
  cv_bridge::CvImageConstPtr image;
  try
  {
    image = cv_bridge::toCvShare( msg, "rgb8" );
  }
  catch ( cv_bridge::Exception &ex )
  {
    NODELET_ERROR( "Could not convert from '%s' to 'bgr8'", msg->encoding.c_str());
  }
  if (camera_model_ == nullptr || camera_model_->cameraInfo().header.frame_id != camera_info->header.frame_id)
  {
    camera_model_.reset( new image_geometry::PinholeCameraModel());
    camera_model_->fromCameraInfo( camera_info );
  }
  cv::Point2d center;
  double radius;

  findOuterCircle( image->image, 1, center, radius, debug_info_ );

  geometry_msgs::PoseStamped pose_2d;
  pose_2d.header.frame_id = image->header.frame_id;
  pose_2d.header.stamp = image->header.stamp;
  pose_2d.pose.position.x = center.x;
  pose_2d.pose.position.y = center.y;
  pose_2d.pose.position.z = radius;
  pose_2d.pose.orientation.w = 1;
  pose_2d_publisher.publish( pose_2d );

  // K[0] is top left entry of K matrix which is focal length in x which is the pixel to meter ratio
  double distance = camera_info->K[0] * circle_radius_ / radius;
  cv::Point3d ray = camera_model_->projectPixelTo3dRay(center);
  ray = ray * distance / cv::norm(ray);
  geometry_msgs::PoseStamped pose;
  pose.header.frame_id = image->header.frame_id;
  pose.header.stamp = image->header.stamp;
  pose.pose.position.x = ray.x;
  pose.pose.position.y = ray.y;
  pose.pose.position.z = ray.z;
  pose.pose.orientation.w = 1;
  pose_publisher_.publish( pose );
}

void PipeDetectionNodelet::enableCallback()
{
  int count = subscriber_count_++;
  NODELET_INFO_STREAM("Subscriber count " << count);
  if ( count == 0 )
  {
    NODELET_INFO( "Starting detection." );
    image_subscriber_ = image_transport_->subscribeCamera( "image", 1, &PipeDetectionNodelet::detectCallback, this );
  }
}

void PipeDetectionNodelet::disableCallback()
{
  int count = subscriber_count_--;
  if ( count == 1 )
  {
    NODELET_INFO( "Stopping detection." );
    image_subscriber_.shutdown();
  }
}
}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(hector_pipe_detection::PipeDetectionNodelet, nodelet::Nodelet)
