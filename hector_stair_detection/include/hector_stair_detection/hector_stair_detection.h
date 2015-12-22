#ifndef Hector_Stair_Detection_H
#define Hector_Stair_Detection_H

#include <ros/ros.h>

#include <tf/tf.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/point_representation.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <std_msgs/Header.h>
#include <nav_msgs/MapMetaData.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>
#include <stdio.h>
#include <string>
#include <math.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <pcl/filters/voxel_grid.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl/surface/processing.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

//#include <vigir_perception_msgs/PointCloudRegionRequest.h>
//#include <vigir_perception_msgs/EnvironmentRegionRequest.h>

#include <hector_stair_detection/hector_stair_detection.h>

#include <hector_stair_detection/BorderAndOrientationOfStairs.h>
#include <hector_stair_detection/PositionAndOrientaion.h>

namespace hector_stair_detection{

class HectorStairDetection{

public:
    HectorStairDetection();
    virtual ~HectorStairDetection();
    void PclCallback(const sensor_msgs::PointCloud2::ConstPtr& pc_msg);

protected:
    ros::Publisher possible_stairs_cloud_pub_;
    ros::Publisher points_on_line_cloud_debug_;
    ros::Publisher surfaceCloud_pub_debug_;
    ros::Publisher final_stairs_cloud_pub_;
    ros::Publisher border_of_stairs_pub_;
    ros::Publisher stairs_position_and_orientaion_pub_;
    ros::Publisher border_and_orientation_stairs_combined_pub_;
    ros::Publisher stairs_position_and_orientaion_with_direction_pub_;

    ros::Subscriber pcl_sub;
    tf::TransformListener listener_;
    Eigen::Affine3d to_map_;

private:
    //params
    double passThroughZMin_;
    double passThroughZMax_;
    double voxelGridX_;
    double voxelGridY_;
    double voxelGridZ_;
    int minRequiredPointsOnLine_;
    double distanceToLineTresh_;
    bool refineSurfaceRequired_; //in hector setup true
    std::string worldFrame_;
    double clusterHeightTresh_;
    double maxClusterXYDimension_;
    double clusterTolerance_;
    int clusterMinSize_;
    int clusterMaxSize_;
    std::string setup_;
    double distTrashSegmentation_;
    double maxDistBetweenStairsPoints_;
    double minHightDistBetweenAllStairsPoints_;

    void getPreprocessedCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud, pcl::PointCloud<pcl::PointNormal>::Ptr &output_cloud);
    void refineOrientaion(Eigen::Vector2f directionStairs, Eigen::Vector2f minXminY, Eigen::Vector2f maxXminY, Eigen::Vector2f minXmaxY, geometry_msgs::PoseStamped &position_and_orientaion);
//    void getFinalStairsCloud_and_position(std::string frameID, Eigen::Vector3f directionS, pcl::PointCloud<pcl::PointXYZ>::Ptr &allPlaneCloud, std::vector<pcl::PointIndices> cluster_indices, std::vector<int> final_cluster_idx,
//                                          pcl::PointCloud<pcl::PointXYZ>::Ptr &final_stairsCloud, visualization_msgs::MarkerArray &stairs_boarder_marker);
    void getFinalStairsCloud_and_position(std::string frameID, Eigen::Vector3f directionS, pcl::PointCloud<pcl::PointNormal>::Ptr &input_surface_cloud, pcl::IndicesClustersPtr cluster_indices, std::vector<int> final_cluster_idx,
                                                                pcl::PointCloud<pcl::PointXYZ>::Ptr &final_stairsCloud, visualization_msgs::MarkerArray &stairs_boarder_marker);
    void getStairsPositionAndOrientation(Eigen::Vector3f base, Eigen::Vector3f point, std::string frameID, Eigen::Vector3f &direction, geometry_msgs::PoseStamped &position_and_orientaion);
    //          void publishResults(pcl::PointCloud<pcl::PointXYZ>::Ptr &input_surface_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &allPlaneCloud,
    //                              std::vector<pcl::PointIndices> cluster_indices, std::vector<int> cluster_idx_corresponding_to_avg_point, Eigen::Vector3f base, Eigen::Vector3f point);
    void publishResults(pcl::PointCloud<pcl::PointNormal>::Ptr &input_surface_cloud,
                                              pcl::IndicesClustersPtr cluster_indices, std::vector<int> final_cluster_idx, Eigen::Vector3f base, Eigen::Vector3f point);
    int getZComponent(Eigen::Vector2f directionStairs, Eigen::Vector2f minXminY, Eigen::Vector2f maxXminY, Eigen::Vector2f minXmaxY);
    float maxDistBetweenPoints(pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud);
    bool pointInCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, pcl::PointXYZ point);
    float minHightDistBetweenPoints(pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud);
    //          void HectorStairDetection::getPointCloudBoundary(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, geometry_msgs::Vector3& min, geometry_msgs::Vector3& max);
};
}

#endif
