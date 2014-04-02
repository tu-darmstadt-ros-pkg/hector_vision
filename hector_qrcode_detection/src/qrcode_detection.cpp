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

#include <hector_qrcode_detection/qrcode_detection.h>
#include <hector_worldmodel_msgs/ImagePercept.h>

#include <cv.h>
#include <cv_bridge/cv_bridge.h>
#include <zbar.h>

using namespace zbar;

namespace hector_qrcode_detection {

qrcode_detection_impl::qrcode_detection_impl(ros::NodeHandle nh, ros::NodeHandle priv_nh)
  : nh_(nh)
  , image_transport_(nh_)
  , listener_(0)
{
  scanner_ = new zbar::ImageScanner;
  scanner_->set_config(ZBAR_QRCODE, ZBAR_CFG_ENABLE, 1);

  rotation_image_size_ = 2;
  priv_nh.getParam("rotation_source_frame", rotation_source_frame_id_);
  priv_nh.getParam("rotation_target_frame", rotation_target_frame_id_);
  priv_nh.getParam("rotation_image_size", rotation_image_size_);

  percept_publisher_ = nh_.advertise<hector_worldmodel_msgs::ImagePercept>("image_percept", 10);
  qrcode_image_publisher_ = image_transport_.advertiseCamera("image/qrcode", 10);
  camera_subscriber_ = image_transport_.subscribeCamera("image", 10, &qrcode_detection_impl::imageCallback, this);

  if (!rotation_target_frame_id_.empty()) {
    listener_ = new tf::TransformListener();
    rotated_image_publisher_ = image_transport_.advertiseCamera("image/rotated", 10);
  }

  ROS_INFO("Successfully initialized the zbar qrcode detector for image %s", camera_subscriber_.getTopic().c_str());
}

qrcode_detection_impl::~qrcode_detection_impl()
{
  delete listener_;
}

void qrcode_detection_impl::imageCallback(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::CameraInfoConstPtr& camera_info)
{
  cv_bridge::CvImageConstPtr cv_image;
  cv_image = cv_bridge::toCvShare(image, "mono8");
  cv::Mat rotation_matrix = cv::Mat::eye(2,3,CV_32FC1);
  double rotation_angle = 0.0;

  ROS_DEBUG("Received new image with %u x %u pixels.", image->width, image->height);

  if (!rotation_target_frame_id_.empty() && listener_) {
    tf::StampedTransform transform;
    std::string source_frame_id_ = rotation_source_frame_id_.empty() ? image->header.frame_id : rotation_source_frame_id_;
    try
    {
      listener_->waitForTransform(rotation_target_frame_id_, source_frame_id_, image->header.stamp, ros::Duration(1.0));
      listener_->lookupTransform(rotation_target_frame_id_, source_frame_id_, image->header.stamp, transform);
    } catch (tf::TransformException& e) {
      ROS_ERROR("%s", e.what());
      return;
    }

    // calculate rotation angle
    tfScalar roll, pitch, yaw;
    transform.getBasis().getRPY(roll, pitch, yaw);
    rotation_angle = -roll;

    // Transform the image.
    try
    {
      cv::Mat in_image = cv_image->image;

      // Compute the output image size.
      int max_dim = in_image.cols > in_image.rows ? in_image.cols : in_image.rows;
      int min_dim = in_image.cols < in_image.rows ? in_image.cols : in_image.rows;
      int noblack_dim = min_dim / sqrt(2);
      int diag_dim = sqrt(in_image.cols*in_image.cols + in_image.rows*in_image.rows);
      int out_size;
      int candidates[] = { noblack_dim, min_dim, max_dim, diag_dim, diag_dim }; // diag_dim repeated to simplify limit case.
      int step = rotation_image_size_;
      out_size = candidates[step] + (candidates[step + 1] - candidates[step]) * (rotation_image_size_ - step);
      //ROS_INFO("out_size: %d", out_size);

      // Compute the rotation matrix.
      rotation_matrix = cv::getRotationMatrix2D(cv::Point2f(in_image.cols / 2.0, in_image.rows / 2.0), 180 * rotation_angle / M_PI, 1);
      rotation_matrix.at<double>(0, 2) += (out_size - in_image.cols) / 2.0;
      rotation_matrix.at<double>(1, 2) += (out_size - in_image.rows) / 2.0;

      // Do the rotation
      cv_bridge::CvImage *temp = new cv_bridge::CvImage(*cv_image);
      cv::warpAffine(in_image, temp->image, rotation_matrix, cv::Size(out_size, out_size));
      cv_image.reset(temp);

      if (rotated_image_publisher_.getNumSubscribers() > 0) {
        sensor_msgs::Image rotated_image;
        cv_image->toImageMsg(rotated_image);
        rotated_image_publisher_.publish(rotated_image, *camera_info);
      }
    }
    catch (cv::Exception &e)
    {
      ROS_ERROR("Image processing error: %s %s %s %i", e.err.c_str(), e.func.c_str(), e.file.c_str(), e.line);
      return;
    }
  }

  // wrap image data
  Image zbar(cv_image->image.cols, cv_image->image.rows, "Y800", cv_image->image.data, cv_image->image.cols * cv_image->image.rows);

  // scan the image for barcodes
  scanner_->scan(zbar);

  // extract results
  hector_worldmodel_msgs::ImagePercept percept;
  percept.header = image->header;
  percept.camera_info = *camera_info;
  percept.info.class_id = "qrcode";
  percept.info.class_support = 1.0;

  for(Image::SymbolIterator symbol = zbar.symbol_begin(); symbol != zbar.symbol_end(); ++symbol) {
    ROS_DEBUG_STREAM("Decoded " << symbol->get_type_name() << " symbol \"" << symbol->get_data() << '"');

    // percept.info.object_id = ros::this_node::getName() + "/" + symbol->get_data();
    //percept.info.object_id = symbol->get_data();
    percept.info.object_support = 1.0;
    percept.info.name = symbol->get_data();

    if (symbol->get_location_size() != 4) {
      ROS_WARN("Could not get symbol locations(location_size != 4)");
      continue;
    }

    // point order is left/top, left/bottom, right/bottom, right/top
    int min_x = 99999999, min_y = 99999999, max_x = 0, max_y = 0;
    for(int i = 0; i < 4; ++i) {
      if (symbol->get_location_x(i) > max_x) max_x = symbol->get_location_x(i);
      if (symbol->get_location_x(i) < min_x) min_x = symbol->get_location_x(i);
      if (symbol->get_location_y(i) > max_y) max_y = symbol->get_location_y(i);
      if (symbol->get_location_y(i) < min_y) min_y = symbol->get_location_y(i);
    }

    // rotate the percept back
    cv::Vec3f left_top_corner(min_x, min_y, 1.0f);
    cv::Vec3f right_bottom_corner(max_x, max_y, 1.0f);

    // TODO: calculate the inverse transformation of rotation_matrix
    if (rotation_angle != 0.0) {
      ROS_ERROR("Non-zero rotations are currently not supported!");
      continue;
    }

    percept.x      = (left_top_corner(0) + right_bottom_corner(0)) / 2;
    percept.y      = (left_top_corner(1) + right_bottom_corner(1)) / 2;
    percept.width  = right_bottom_corner(0) - left_top_corner(0);
    percept.height = right_bottom_corner(1) - left_top_corner(1);
    percept_publisher_.publish(percept);

//    ROS_DEBUG("location: min_x: %d  min_y: %d  max_x: %d  max_y: %d", min_x, min_y, max_x, max_y);
//    ROS_DEBUG("rotated:  min_x: %f  min_y: %f  max_x: %f  max_y: %f", left_top_corner(0), left_top_corner(1), right_bottom_corner(0), right_bottom_corner(1));
//    ROS_DEBUG("percept:  x: %f  y: %f  width: %f  height: %f", percept.x, percept.y, percept.width, percept.height);

    if (qrcode_image_publisher_.getNumSubscribers() > 0) {
      try {
        cv::Rect rect(cv::Point2i(std::max(min_x, 0), std::max(min_y, 0)), cv::Point2i(std::min(max_x, cv_image->image.cols), std::min(max_y, cv_image->image.rows)));
        cv_bridge::CvImagePtr qrcode_cv(new cv_bridge::CvImage(*cv_image));
        qrcode_cv->image = cv_image->image(rect);

        sensor_msgs::Image qrcode_image;
        qrcode_cv->toImageMsg(qrcode_image);
        qrcode_image_publisher_.publish(qrcode_image, *camera_info);
      } catch(cv::Exception& e) {
        ROS_ERROR("cv::Exception: %s", e.what());
      }
    }
  }

  // clean up
  zbar.set_data(NULL, 0);
}

} // namespace hector_qrcode_detection
