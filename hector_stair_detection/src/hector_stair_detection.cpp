#include <ros/ros.h>
#include <hector_stair_detection/hector_stair_detection.h>

namespace hector_stair_detection{
HectorStairDetection::HectorStairDetection(){
    ROS_INFO ("HectorStairDetection started");
    ros::NodeHandle nh("");

    //load params
    nh.param("passThroughZMin", passThroughZMin_, 0.1);
    nh.param("passThroughZMax", passThroughZMax_, 2.5);
    nh.param("voxelGridX", voxelGridX_, 0.04);
    nh.param("voxelGridY", voxelGridY_, 0.04);
    nh.param("voxelGridZ", voxelGridZ_, 0.04);
    nh.param("minRequiredPointsOnLine", minRequiredPointsOnLine_, 3);
    nh.param("distanceToLineTresh", distanceToLineTresh_, 0.1);
    nh.param("refineSurfaceRequired", refineSurfaceRequired_, false);
    nh.param("worldFrame", worldFrame_, std::string("/world")); //has to be true in hector-setup
    nh.param("culsterHeightTresh", clusterHeightTresh_, 0.1);
    nh.param("clusterTolerance", clusterTolerance_, 0.075);
    nh.param("clusterMinSize", clusterMinSize_, 50);
    nh.param("clusterMaxSize", clusterMaxSize_, 200);
    nh.param("setup", setup_, std::string("argo"));
    nh.param("maxClusterXYDimension", maxClusterXYDimension_, 1.0);
    nh.param("minHightDistBetweenAllStairsPoints", minHightDistBetweenAllStairsPoints_, 0.2);
    nh.param("maxDistBetweenStairsPoints", maxDistBetweenStairsPoints_, 3.0);
    nh.param("planeSegDistTresh", planeSegDistTresh_, 0.01);
    nh.param("planeSegAngleEps", planeSegAngleEps_, 0.1);
    nh.param("hesseTresh", hesseTresh_, 0.1);

    possible_stairs_cloud_pub_= nh.advertise<pcl::PointCloud<pcl::PointXYZI> >("/hector_stair_detection/possible_stairs_cloud", 100, true);
    points_on_line_cloud_debug_= nh.advertise<pcl::PointCloud<pcl::PointXYZI> >("/hector_stair_detection/point_on_line_debug", 100, true);
    surfaceCloud_pub_debug_= nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("/hector_stair_detection/surfaceCloud_debug", 100, true);
    final_stairs_cloud_pub_=  nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("/hector_stair_detection/final_stairs_cloud", 100, true);
    border_of_stairs_pub_= nh.advertise<visualization_msgs::MarkerArray>("/hector_stair_detection/boarder_of_stairs", 100, true);
    stairs_position_and_orientaion_pub_= nh.advertise<geometry_msgs::PoseStamped>("/hector_stair_detection/stairs_orientation", 100, true);
    stairs_position_and_orientaion_with_direction_pub_= nh.advertise<hector_stair_detection_msgs::PositionAndOrientaion>("/hector_stair_detection/stairs_orientaion_as_vector", 100, true);
    border_and_orientation_stairs_combined_pub_= nh.advertise<hector_stair_detection_msgs::BorderAndOrientationOfStairs>("/hector_stair_detection/border_and_orientation_of_stairs", 100, true);
    cloud_after_plane_detection_debug_pub_= nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("/hector_stair_detection/cloud_after_plane_detection_debug_", 100, true);
    line_marker_pub_= nh.advertise<visualization_msgs::Marker>("/hector_stair_detection/stairs_normal", 100, true);

    temp_orginal_pub_=nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("/hector_stair_detection/temp_original", 100, true);
    temp_after_pass_trough_pub_=nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("/hector_stair_detection/temp_after_pass_trough", 100, true);
    temp_after_voxel_grid_pub_=nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("/hector_stair_detection/temp_after_voxel_grid", 100, true);
    temp_after_mls_pub_=nh.advertise<pcl::PointCloud<pcl::PointNormal> >("/hector_stair_detection/temp_after_mls", 100, true);



    if(setup_.compare(std::string("argo"))==0){
        //TODO:: get pointcloud from Service
        //        pointcloud_srv_client_ = nh.serviceClient<vigir_perception_msgs::PointCloudRegionRequest>("/flor/worldmodel/pointcloud_roi");
        //        vigir_perception_msgs::PointCloudRegionRequest srv;
        //        vigir_perception_msgs::EnvironmentRegionRequest erreq;
        //        erreq.header=worldFrame_;
        //        erreq.bounding_box_max.x=;
        //        erreq.bounding_box_max.y=;
        //        erreq.bounding_box_max.z=passThroughZMax_;
        //        erreq.bounding_box_min.x=;
        //        erreq.bounding_box_min.y=;
        //        erreq.bounding_box_min.z=passThroughZMin_;
        //        erreq.resolution=voxelGridX_;
        //        erreq.request_augment=0;
        //        srv.request.region_req=erreq;
        //        srv.request.aggregation_size=500;
        //        if(!pointcloud_srv_client_.call(srv)){
        //            ROS_ERROR("/flor/worldmodel/pointcloud_roi is not working");
        //        }else{
        //            sensor_msgs::PointCloud2 pointCloud_world;
        //            pointCloud_world=srv.response.cloud;
        pcl_sub = nh.subscribe("/worldmodel_main/pointcloud_vis", 10, &HectorStairDetection::PclCallback, this);
    }else{
        //        pcl_sub = nh.subscribe("/openni/depth/points", 1, &HectorStairDetection::PclCallback, this);
        //        pcl_sub = nh.subscribe("/hector_octomap_server/octomap_point_cloud_centers", 1, &HectorStairDetection::PclCallback, this);
        pcl_sub = nh.subscribe("/hector_aggregate_cloud/aggregated_cloud", 1, &HectorStairDetection::PclCallback, this);
    }

}

HectorStairDetection::~HectorStairDetection()
{}

void HectorStairDetection::publishResults(pcl::PointCloud<pcl::PointNormal>::Ptr &input_surface_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &planeCloud,
                                          pcl::IndicesClustersPtr cluster_indices, std::vector<int> final_cluster_idx, Eigen::Vector3f base, Eigen::Vector3f point){

    geometry_msgs::PoseStamped position_and_orientaion;
    Eigen::Vector3f directionStairs;
    getStairsPositionAndOrientation(base, point, input_surface_cloud->header.frame_id, directionStairs, position_and_orientaion);
    hector_stair_detection_msgs::PositionAndOrientaion pos_and_orientaion_message;
    pos_and_orientaion_message.orientation_of_stairs=position_and_orientaion;
    pos_and_orientaion_message.directionX=directionStairs(0);
    pos_and_orientaion_message.directionY=directionStairs(1);
    pos_and_orientaion_message.directionZ=directionStairs(2);

    //compute and pubish max and min boarder of the stairs
    pcl::PointCloud<pcl::PointXYZ>::Ptr final_stairsCloud(new pcl::PointCloud<pcl::PointXYZ>);
    visualization_msgs::MarkerArray stairs_boarder_marker;
    getFinalStairsCloud_and_position(input_surface_cloud->header.frame_id, directionStairs, input_surface_cloud, planeCloud, cluster_indices, final_cluster_idx, final_stairsCloud, stairs_boarder_marker, base);

    if(final_stairs_cloud_pub_.getNumSubscribers()>0){
        final_stairs_cloud_pub_.publish(final_stairsCloud);
    }
    if(border_of_stairs_pub_.getNumSubscribers()>0){
        border_of_stairs_pub_.publish(stairs_boarder_marker);
    }

    hector_stair_detection_msgs::BorderAndOrientationOfStairs border_and_orientation_msg;
    border_and_orientation_msg.header.frame_id=input_surface_cloud->header.frame_id;
    border_and_orientation_msg.border_of_stairs= stairs_boarder_marker;
    border_and_orientation_msg.orientation_of_stairs=position_and_orientaion;
    border_and_orientation_msg.number_of_points= final_cluster_idx.size();
    border_and_orientation_msg.directionX=directionStairs(0);
    border_and_orientation_msg.directionY=directionStairs(1);
    border_and_orientation_msg.directionZ=directionStairs(2);

    if(border_and_orientation_stairs_combined_pub_.getNumSubscribers()>0){
        border_and_orientation_stairs_combined_pub_.publish(border_and_orientation_msg);
    }

    if(stairs_position_and_orientaion_with_direction_pub_.getNumSubscribers()>0){
        stairs_position_and_orientaion_with_direction_pub_.publish(pos_and_orientaion_message);
    }
    if(stairs_position_and_orientaion_pub_.getNumSubscribers()>0){
        stairs_position_and_orientaion_pub_.publish(position_and_orientaion);
    }
}

void HectorStairDetection::getStairsPositionAndOrientation(Eigen::Vector3f &base, Eigen::Vector3f point, std::string frameID, Eigen::Vector3f &direction, geometry_msgs::PoseStamped &position_and_orientaion){
    Eigen::Vector3f stairs_position= 0.5*(base + point);

    position_and_orientaion.header.frame_id=frameID;
    position_and_orientaion.pose.position.x=stairs_position(0);
    position_and_orientaion.pose.position.y=stairs_position(1);
    position_and_orientaion.pose.position.z=stairs_position(2);

    if(point(2) >=base(2)){
        direction=point-base;
    }else{
        direction=base-point;
        base=point;
    }
    float stairs_yaw= atan2(direction(1), direction(0));
    float staris_pitch= atan2(direction(1)*sin(stairs_yaw)+direction(0)*cos(stairs_yaw), direction(2))+M_PI_2;
    tf::Quaternion temp;
    temp.setEulerZYX(stairs_yaw,staris_pitch,0.0);
    position_and_orientaion.pose.orientation.x=temp.getX();
    position_and_orientaion.pose.orientation.y=temp.getY();
    position_and_orientaion.pose.orientation.z=temp.getZ();
    position_and_orientaion.pose.orientation.w=temp.getW();
}

void HectorStairDetection::getFinalStairsCloud_and_position(std::string frameID, Eigen::Vector3f directionS, pcl::PointCloud<pcl::PointNormal>::Ptr &input_surface_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &planeCloud, pcl::IndicesClustersPtr cluster_indices, std::vector<int> final_cluster_idx,
                                                            pcl::PointCloud<pcl::PointXYZ>::Ptr &final_stairsCloud, visualization_msgs::MarkerArray &stairs_boarder_marker, Eigen::Vector3f base){
    int cluster_counter=0;
    float minZ= FLT_MAX;
    float maxZ= -FLT_MAX;
    int minZIndex=-1;
    int maxZIndex=-1;
    final_stairsCloud->header.frame_id=frameID;
//    make use of the plane cloud
//    if(planeCloud->size() !=0){
//        Eigen::Vector3f orthogonalDir=directionS.cross(Eigen::Vector3f(0,0,1));
//        Eigen::Vector3f point;
//        pcl::PointCloud<pcl::PointXYZ>::Ptr testCloud(new pcl::PointCloud<pcl::PointXYZ>());
//        testCloud->header.frame_id=frameID;
//        for(int i=0; i<planeCloud->size(); i++){
//            point(0)=planeCloud->points.at(i).x;
//            point(1)=planeCloud->points.at(i).y;
//            point(2)=planeCloud->points.at(i).z;
//            //TODO:: check angle
////            float angle=acos((directionS.dot(point-base))/(directionS.norm()*(point-base).norm()));
//            float angle=acos((orthogonalDir.dot(point-base))/(orthogonalDir.norm()*(point-base).norm()));

//            ROS_INFO("ange: %f", angle);
//            if(angle < M_PI_2){
//                testCloud->push_back(planeCloud->points.at(i));
//            }

//            //compute distance
//            float tempA= std::sqrt(directionS.cross(point-base).dot(directionS.cross(point-base)));
//            float lengthDirection= std::sqrt(directionS.dot(directionS));
//            float distance= tempA/lengthDirection;

//            //save max depening on sign


//        }
//        cloud_after_plane_detection_debug_pub_.publish(testCloud);
//    }

    for (int i = 0; i < cluster_indices->size (); ++i){
        std::vector<int>::iterator idx_it;

        idx_it = find (final_cluster_idx.begin(), final_cluster_idx.end(), cluster_counter);
        if (idx_it != final_cluster_idx.end()){
            for (int j = 0; j < (*cluster_indices)[i].indices.size (); ++j){
                pcl::PointXYZ tempP;
                tempP.x=input_surface_cloud->points[(*cluster_indices)[i].indices[j]].x;
                tempP.y=input_surface_cloud->points[(*cluster_indices)[i].indices[j]].y;
                tempP.z=input_surface_cloud->points[(*cluster_indices)[i].indices[j]].z;
                final_stairsCloud->points.push_back(tempP);

                if(tempP.z < minZ){
                    minZ=tempP.z;
                    minZIndex=i;
                }

                if(tempP.z > maxZ){
                    maxZ=tempP.z;
                    maxZIndex=i;
                }
            }
        }
        cluster_counter=cluster_counter+1;
    }

    Eigen::Vector3f minStepAVG;
    minStepAVG(0)=0;
    minStepAVG(1)=0;
    minStepAVG(2)=0;
    Eigen::Vector3f maxStepAVG;
    maxStepAVG(0)=0;
    maxStepAVG(1)=0;
    maxStepAVG(2)=0;


    for (int j = 0; j < (*cluster_indices)[minZIndex].indices.size (); ++j){
        minStepAVG(0)=minStepAVG(0)+input_surface_cloud->points[(*cluster_indices)[minZIndex].indices[j]].x;
        minStepAVG(1)=minStepAVG(1)+input_surface_cloud->points[(*cluster_indices)[minZIndex].indices[j]].y;
        minStepAVG(2)=minStepAVG(2)+input_surface_cloud->points[(*cluster_indices)[minZIndex].indices[j]].z;
    }

    minStepAVG=minStepAVG/(*cluster_indices)[minZIndex].indices.size ();

    for (int j = 0; j < (*cluster_indices)[maxZIndex].indices.size (); ++j){
        maxStepAVG(0)=maxStepAVG(0)+input_surface_cloud->points[(*cluster_indices)[maxZIndex].indices[j]].x;
        maxStepAVG(1)=maxStepAVG(1)+input_surface_cloud->points[(*cluster_indices)[maxZIndex].indices[j]].y;
        maxStepAVG(2)=maxStepAVG(2)+input_surface_cloud->points[(*cluster_indices)[maxZIndex].indices[j]].z;
    }

    maxStepAVG=maxStepAVG/(*cluster_indices)[maxZIndex].indices.size ();


    for(int i=0; i<4; i++){
        visualization_msgs::Marker marker;
        marker.header.frame_id = frameID;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.ns = "hector_stair_detection";
        marker.id = i;
        if((directionS(0)> 0 && directionS(1) >0) || (directionS(0)< 0 && directionS(1) <0)){
            if(i==0){
                marker.pose.position.x = minStepAVG(0) - 0.5*sin(atan2(directionS(1), directionS(0)));
                marker.pose.position.y = minStepAVG(1) + 0.5*cos(atan2(directionS(1), directionS(0)));
                marker.pose.position.z=minStepAVG(2);
            }

            if(i==1){
                marker.pose.position.x = minStepAVG(0) + 0.5*sin(atan2(directionS(1), directionS(0)));
                marker.pose.position.y = minStepAVG(1) - 0.5*cos(atan2(directionS(1), directionS(0)));
                marker.pose.position.z=minStepAVG(2);
            }

            if(i==2){
                marker.pose.position.x = maxStepAVG(0) - 0.5*sin(atan2(directionS(1), directionS(0)));
                marker.pose.position.y = maxStepAVG(1) + 0.5*cos(atan2(directionS(1), directionS(0)));
                marker.pose.position.z=maxStepAVG(2);
            }

            if(i==3){
                marker.pose.position.x = maxStepAVG(0) + 0.5*sin(atan2(directionS(1), directionS(0)));
                marker.pose.position.y = maxStepAVG(1) - 0.5*cos(atan2(directionS(1), directionS(0)));
                marker.pose.position.z=maxStepAVG(2);
            }
        }else{
            if(i==0){
                marker.pose.position.x = minStepAVG(0) + 0.5*sin(atan2(directionS(1), directionS(0)));
                marker.pose.position.y = minStepAVG(1) + 0.5*cos(atan2(directionS(1), directionS(0)));
                marker.pose.position.z=minStepAVG(2);
            }

            if(i==1){
                marker.pose.position.x = minStepAVG(0) - 0.5*sin(atan2(directionS(1), directionS(0)));
                marker.pose.position.y = minStepAVG(1) - 0.5*cos(atan2(directionS(1), directionS(0)));
                marker.pose.position.z=minStepAVG(2);
            }

            if(i==2){
                marker.pose.position.x = maxStepAVG(0) + 0.5*sin(atan2(directionS(1), directionS(0)));
                marker.pose.position.y = maxStepAVG(1) + 0.5*cos(atan2(directionS(1), directionS(0)));
                marker.pose.position.z=maxStepAVG(2);
            }

            if(i==3){
                marker.pose.position.x = maxStepAVG(0) - 0.5*sin(atan2(directionS(1), directionS(0)));
                marker.pose.position.y = maxStepAVG(1) - 0.5*cos(atan2(directionS(1), directionS(0)));
                marker.pose.position.z=maxStepAVG(2);
            }
        }

        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.2;
        marker.scale.y = 0.2;
        marker.scale.z = 0.2;
        marker.color.a = 1.0;
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;

        stairs_boarder_marker.markers.push_back(marker);

    }
        projectStairsToFloor(directionS, stairs_boarder_marker);
}

void HectorStairDetection::projectStairsToFloor(Eigen::Vector3f direction, visualization_msgs::MarkerArray &stairs_boarder_marker){
    float first_step_z=0.05;
    float minZ= FLT_MAX;
    int minPos1;
    int minPos2;
    for(int i=0; i<stairs_boarder_marker.markers.size(); i++){
        if(stairs_boarder_marker.markers.at(i).pose.position.z <minZ){
            minZ=stairs_boarder_marker.markers.at(i).pose.position.z;
            minPos1=i;
        }
    }

    minZ= FLT_MAX;
    for(int i=0; i<stairs_boarder_marker.markers.size(); i++){
        if(i != minPos1 && stairs_boarder_marker.markers.at(i).pose.position.z <minZ){
            minZ=stairs_boarder_marker.markers.at(i).pose.position.z;
            minPos2=i;
        }
    }

    stairs_boarder_marker.markers.at(minPos1).pose.position.x=stairs_boarder_marker.markers.at(minPos1).pose.position.x+(first_step_z-stairs_boarder_marker.markers.at(minPos1).pose.position.z)/direction(2)*direction(0);
    stairs_boarder_marker.markers.at(minPos1).pose.position.y=stairs_boarder_marker.markers.at(minPos1).pose.position.y+(first_step_z-stairs_boarder_marker.markers.at(minPos1).pose.position.z)/direction(2)*direction(1);
    stairs_boarder_marker.markers.at(minPos1).pose.position.z=first_step_z;
    stairs_boarder_marker.markers.at(minPos2).pose.position.x=stairs_boarder_marker.markers.at(minPos2).pose.position.x+(first_step_z-stairs_boarder_marker.markers.at(minPos2).pose.position.z)/direction(2)*direction(0);
    stairs_boarder_marker.markers.at(minPos2).pose.position.y=stairs_boarder_marker.markers.at(minPos2).pose.position.y+(first_step_z-stairs_boarder_marker.markers.at(minPos2).pose.position.z)/direction(2)*direction(1);
    stairs_boarder_marker.markers.at(minPos2).pose.position.z=first_step_z;
}

int HectorStairDetection::getZComponent(Eigen::Vector2f directionStairs, Eigen::Vector2f minXminY, Eigen::Vector2f maxXminY, Eigen::Vector2f minXmaxY){
    Eigen::Vector2f direction1;
    Eigen::Vector2f direction2;
    Eigen::Vector2f direction3;
    Eigen::Vector2f direction4;

    direction1=maxXminY-minXminY;
    direction2=minXmaxY-minXminY;
    direction3=minXminY-maxXminY;
    direction4=minXminY-minXmaxY;

    float angle1=acos((directionStairs.dot(direction1))/(directionStairs.norm()*direction1.norm()));
    float angle2=acos((directionStairs.dot(direction2))/(directionStairs.norm()*direction2.norm()));
    float angle3=acos((directionStairs.dot(direction3))/(directionStairs.norm()*direction3.norm()));
    float angle4=acos((directionStairs.dot(direction4))/(directionStairs.norm()*direction4.norm()));

    if(angle1>=angle2 && angle1>=angle3 && angle1>=angle4){
        return 1;
    }

    if(angle2>=angle1 && angle2>=angle3 && angle2>=angle4){
        return 2;
    }

    if(angle3>=angle2 && angle3>=angle1 && angle3>=angle4){
        return 3;
    }

    if(angle4>=angle2 && angle4>=angle3 && angle4>=angle1){
        return 4;
    }
}

void HectorStairDetection::getPreprocessedCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud, pcl::PointCloud<pcl::PointNormal>::Ptr &output_cloud){
    ROS_INFO("Hector Stair Detection get Surface");
    pcl::PointCloud<pcl::PointXYZ>::Ptr processCloud_v1(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr processCloud_v2(new pcl::PointCloud<pcl::PointXYZ>());

    temp_orginal_pub_.publish(input_cloud);

    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(input_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(passThroughZMin_, passThroughZMax_);
    pass.filter(*processCloud_v1);

//            pass.setInputCloud(processCloud_v1);
//            pass.setFilterFieldName("y");
//            pass.setFilterLimits(-1.0, 1.0);
//            pass.filter(*processCloud_v1);

//            pass.setInputCloud(processCloud_v1);
//            pass.setFilterFieldName("x");
//            pass.setFilterLimits(0.0, 3.5);
//            pass.filter(*processCloud_v1);

    temp_after_pass_trough_pub_.publish(processCloud_v1);

    pcl::VoxelGrid<pcl::PointXYZ> vox;
    vox.setInputCloud(processCloud_v1);
    vox.setLeafSize(voxelGridX_, voxelGridY_, voxelGridZ_);
    vox.setDownsampleAllData(false);
    vox.filter(*processCloud_v2);

    temp_after_voxel_grid_pub_.publish(processCloud_v2);

    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
    mls.setSearchRadius(0.05);
    mls.setPolynomialOrder(1);
    mls.setComputeNormals(true);
    mls.setInputCloud(processCloud_v2);

    boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > point_cloud_mls_normal;
    point_cloud_mls_normal.reset(new pcl::PointCloud<pcl::PointNormal>);
    mls.process(*point_cloud_mls_normal);

    temp_after_mls_pub_.publish(point_cloud_mls_normal);

    //    pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
    //    ne.setInputCloud (processCloud_v2);
    //      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    //      ne.setSearchMethod (tree);
    //      // Output datasets
    //      pcl::PointCloud<pcl::PointNormal>::Ptr point_cloud_mls_normal (new pcl::PointCloud<pcl::PointNormal>);
    //      // Use all neighbors in a sphere of radius 3cm
    //      ne.setRadiusSearch (0.03);
    //      // Compute the features
    //      ne.compute (*point_cloud_mls_normal);

    //    temp_after_mls_pub_.publish(point_cloud_mls_normal);

    output_cloud->clear();
    output_cloud->header.frame_id=input_cloud->header.frame_id;

    output_cloud.reset(new pcl::PointCloud<pcl::PointNormal>(*point_cloud_mls_normal));


    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*output_cloud,*output_cloud, indices);
    pcl::removeNaNNormalsFromPointCloud(*output_cloud,*output_cloud, indices);

    if(surfaceCloud_pub_debug_.getNumSubscribers()>0){
        surfaceCloud_pub_debug_.publish(output_cloud);
    }
}

bool customRegionGrowing (const pcl::PointNormal& point_a, const pcl::PointNormal& point_b, float squared_distance){
    Eigen::Map<const Eigen::Vector3f> point_b_normal = point_b.normal;

    Eigen::Vector3f d1(1,0,0);
    Eigen::Vector3f d2(0,1,0);

    if(fabs(d1.dot(point_b_normal)) < 0.2 && fabs(d2.dot(point_b_normal)) < 0.2){
        return true;
    }
    return false;
}


void HectorStairDetection::PclCallback(const sensor_msgs::PointCloud2::ConstPtr& pc_msg){
    ROS_INFO("stairs position callback enterd");
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*pc_msg,pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2,*input_cloud);
    //transform cloud to /world
    tf::StampedTransform transform_cloud_to_map;
    try{
        ros::Time time = pc_msg->header.stamp;
        listener_.waitForTransform(worldFrame_, pc_msg->header.frame_id,
                                   time, ros::Duration(3.0));
        listener_.lookupTransform(worldFrame_, pc_msg->header.frame_id,
                                  time, transform_cloud_to_map);
    }
    catch (tf::TransformException ex){
        ROS_ERROR("Lookup Transform failed: %s",ex.what());
        return;
    }

    tf::transformTFToEigen(transform_cloud_to_map, to_map_);

    // Transform to /world
    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > cloud_tmp(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*input_cloud, *cloud_tmp, to_map_);
    input_cloud = cloud_tmp;
    input_cloud->header.frame_id= transform_cloud_to_map.frame_id_;

    float clusterHeightTresh=clusterHeightTresh_;

    //try find planes with pcl::segmentation
    pcl::PointCloud<pcl::PointNormal>::Ptr input_surface_cloud(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr possible_stairsCloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_cloud_debug(new pcl::PointCloud<pcl::PointXYZI>);

    this->getPreprocessedCloud(input_cloud, input_surface_cloud);

    possible_stairsCloud->header.frame_id=input_surface_cloud->header.frame_id;

    // try conditional clustering
    pcl::IndicesClustersPtr clusters (new pcl::IndicesClusters);
    pcl::ConditionalEuclideanClustering<pcl::PointNormal> cec (false);
    cec.setInputCloud (input_surface_cloud);
    cec.setConditionFunction (&customRegionGrowing);
    cec.setClusterTolerance (clusterTolerance_);
    cec.setMinClusterSize (clusterMinSize_);
    cec.setMaxClusterSize (clusterMaxSize_);
    cec.segment (*clusters);

    //Clustering
    std::vector<Eigen::Vector3f> avg_point_per_cluster;
    float sum_x;
    float sum_y;
    float sum_z;
    float min_x=FLT_MAX;
    float max_x=-FLT_MAX;
    float min_y=FLT_MAX;
    float max_y=-FLT_MAX;
    float min_z=FLT_MAX;
    float max_z=-FLT_MAX;
    int clusterSize;

    int clusterColor=0;
    int clusterCounter=0;

    ROS_INFO("number cluster: %i", clusters->size ());
    std::vector<int> cluster_idx_corresponding_to_avg_point;
    for (int i = 0; i < clusters->size (); ++i){
        clusterSize= (*clusters)[i].indices.size ();
        sum_x=0;
        sum_y=0;
        sum_z=0;
        min_x=FLT_MAX;
        max_x=-FLT_MAX;
        min_y=FLT_MAX;
        max_y=-FLT_MAX;
        min_z=FLT_MAX;
        max_z=-FLT_MAX;
        pcl::PointNormal point;
        for (int j = 0; j < (*clusters)[i].indices.size (); ++j){
            point=input_surface_cloud->points[(*clusters)[i].indices[j]];
            sum_x=sum_x + point.x;
            sum_y=sum_y + point.y;
            sum_z=sum_z + point.z;
            if(min_x > point.x){
                min_x=point.x;
            }
            if(max_x < point.x){
                max_x=point.x;
            }
            if(min_y > point.y){
                min_y=point.y;
            }
            if(max_y < point.y){
                max_y=point.y;
            }
            if(min_z > point.z){
                min_z=point.z;
            }
            if(max_z < point.z){
                max_z=point.z;
            }
        }

        if(fabs(max_x-min_x)>maxClusterXYDimension_ || fabs(max_y-min_y)>maxClusterXYDimension_){
            clusterCounter=clusterCounter+1;
            continue;
        }

        Eigen::Vector3f avg_point;
        avg_point(0)=sum_x/clusterSize;
        avg_point(1)=sum_y/clusterSize;
        avg_point(2)=sum_z/clusterSize;

        if(fabs(max_z-min_z)/2<clusterHeightTresh){
            avg_point_per_cluster.push_back(avg_point);
            cluster_idx_corresponding_to_avg_point.push_back(clusterCounter);

            pcl::PointXYZI tempP;
            tempP.x=avg_point(0);
            tempP.y=avg_point(1);
            tempP.z=avg_point(2);
            tempP.intensity=clusterColor;
            possible_stairsCloud->points.push_back(tempP);
            cluster_cloud_debug->clear();
            cluster_cloud_debug->header.frame_id=worldFrame_;
            for (int j = 0; j < (*clusters)[i].indices.size (); ++j){
                pcl::PointXYZI tempP;
                tempP.x=input_surface_cloud->points[(*clusters)[i].indices[j]].x;
                tempP.y=input_surface_cloud->points[(*clusters)[i].indices[j]].y;
                tempP.z=input_surface_cloud->points[(*clusters)[i].indices[j]].z;
                tempP.intensity=clusterColor;
                possible_stairsCloud->points.push_back(tempP);
                cluster_cloud_debug->points.push_back(tempP);
            }
            clusterColor=clusterColor+1;
        }
        clusterCounter=clusterCounter+1;
    }

    if(possible_stairs_cloud_pub_.getNumSubscribers()>0){
        possible_stairs_cloud_pub_.publish(possible_stairsCloud);
    }
    ROS_INFO("number final cluster: %i", avg_point_per_cluster.size());

    if(avg_point_per_cluster.size()>=2){

        //jeder punkt mit jedem Grade bilden und Abstände berechnen
        int minCountPointsAtLine=minRequiredPointsOnLine_; //number points to stay at line
        int pointsAtLineCounter=0;
        float distanceToLineThreshold=distanceToLineTresh_;
        Eigen::Vector3f direction;
        Eigen::Vector3f base;
        Eigen::Vector3f point;
        std::vector<int> point_in_line_counter;
        point_in_line_counter.resize(avg_point_per_cluster.size());
        for(int i=0; i< point_in_line_counter.size(); i++){
            point_in_line_counter.at(i)=0;
        }


        int maxPointOnLineCounter=0;
        int best_pair_idx1;
        int best_pair_idx2;
        pcl::PointCloud<pcl::PointXYZI>::Ptr debug_line_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        if(avg_point_per_cluster.size() != 0){
            for(int i=0; i<avg_point_per_cluster.size()-1; i++){
                for(int j=i+1; j<avg_point_per_cluster.size(); j++){
                    debug_line_cloud->clear();
                    debug_line_cloud->header.frame_id=input_surface_cloud->header.frame_id;

                    base(0)= avg_point_per_cluster.at(i)(0);
                    base(1)= avg_point_per_cluster.at(i)(1);
                    base(2)= avg_point_per_cluster.at(i)(2);

                    direction(0)= avg_point_per_cluster.at(j)(0) - base(0);
                    direction(1)= avg_point_per_cluster.at(j)(1) - base(1);
                    direction(2)= avg_point_per_cluster.at(j)(2) - base(2);

                    pcl::PointXYZI pushbackPoint;
                    pushbackPoint.x=base(0);
                    pushbackPoint.y=base(1);
                    pushbackPoint.z=base(2);
                    pushbackPoint.intensity=10;
                    debug_line_cloud->push_back(pushbackPoint);
                    pushbackPoint.x=direction(0)+base(0);
                    pushbackPoint.y=direction(1)+base(1);
                    pushbackPoint.z=direction(2)+base(2);
                    pushbackPoint.intensity=10;
                    debug_line_cloud->push_back(pushbackPoint);

                    pointsAtLineCounter=0;
                    for(int p=0; p<avg_point_per_cluster.size(); p++){
                        point(0)=avg_point_per_cluster.at(p)(0);
                        point(1)=avg_point_per_cluster.at(p)(1);
                        point(2)=avg_point_per_cluster.at(p)(2);

                        float tempA= std::sqrt(direction.cross(point-base).dot(direction.cross(point-base)));
                        float lengthDirection= std::sqrt(direction.dot(direction));
                        float distance= tempA/lengthDirection;

                        if(distance <= distanceToLineThreshold){
                            pointsAtLineCounter=pointsAtLineCounter+1;
                            pushbackPoint.x=point(0);
                            pushbackPoint.y=point(1);
                            pushbackPoint.z=point(2);
                            pushbackPoint.intensity=distance;
                            debug_line_cloud->push_back(pushbackPoint);

                            point_in_line_counter.at(p)=point_in_line_counter.at(p)+1;
                        }
                    }

                    if(pointsAtLineCounter >= maxPointOnLineCounter){
                        maxPointOnLineCounter=pointsAtLineCounter;
                        best_pair_idx1=i;
                        best_pair_idx2=j;
                    }



                }
            }
        }

        int iterationCounter=0;
        while(1){
            //get best pair index depending on iteration
            maxPointOnLineCounter=0;
            best_pair_idx1=-1;
            best_pair_idx2=-1;
            for(int p=0; p<point_in_line_counter.size(); p++){
                if(point_in_line_counter.at(p) > maxPointOnLineCounter){
                    maxPointOnLineCounter=point_in_line_counter.at(p);
                    best_pair_idx2=best_pair_idx1;
                    best_pair_idx1=p;
                }
            }

            if(best_pair_idx1==-1 || best_pair_idx2==-1){
                break;
            }

            point_in_line_counter.at(best_pair_idx1)=0;

            //construct line form most used points, middel is the position of the stairs

            base=avg_point_per_cluster.at(best_pair_idx1);

            direction(0)= avg_point_per_cluster.at(best_pair_idx2)(0) - base(0);
            direction(1)= avg_point_per_cluster.at(best_pair_idx2)(1) - base(1);
            direction(2)= avg_point_per_cluster.at(best_pair_idx2)(2) - base(2);

            pcl::PointXYZI pushbackPoint;
            debug_line_cloud->clear();
            debug_line_cloud->header.frame_id=input_cloud->header.frame_id;
            pushbackPoint.x=base(0);
            pushbackPoint.y=base(1);
            pushbackPoint.z=base(2);

            debug_line_cloud->push_back(pushbackPoint);
            pushbackPoint.x=avg_point_per_cluster.at(best_pair_idx2)(0);
            pushbackPoint.y=avg_point_per_cluster.at(best_pair_idx2)(1);
            pushbackPoint.z=avg_point_per_cluster.at(best_pair_idx2)(2);

            debug_line_cloud->push_back(pushbackPoint);

            pointsAtLineCounter=2;
            std::vector<int> final_cluster_idx;
            for(int p=0; p<avg_point_per_cluster.size(); p++){
                if(p==best_pair_idx1 || p== best_pair_idx2) {
                    final_cluster_idx.push_back(cluster_idx_corresponding_to_avg_point.at(p));
                    continue;
                }
                pushbackPoint.x=point(0)=avg_point_per_cluster.at(p)(0);
                pushbackPoint.y=point(1)=avg_point_per_cluster.at(p)(1);
                pushbackPoint.z=point(2)=avg_point_per_cluster.at(p)(2);


                float tempA= std::sqrt(direction.cross(point-base).dot(direction.cross(point-base)));
                float lengthDirection= std::sqrt(direction.dot(direction));
                float distance= tempA/lengthDirection;

                if(distance <= distanceToLineThreshold){
                    pointsAtLineCounter=pointsAtLineCounter+1;
                    final_cluster_idx.push_back(cluster_idx_corresponding_to_avg_point.at(p));
                    debug_line_cloud->push_back(pushbackPoint);

                }
            }
            points_on_line_cloud_debug_.publish(debug_line_cloud);
            if(pointsAtLineCounter >= minCountPointsAtLine){
                //publish results
                if(points_on_line_cloud_debug_.getNumSubscribers()>0){
                    points_on_line_cloud_debug_.publish(debug_line_cloud);
                }
                if(maxDistBetweenPoints(debug_line_cloud) <= maxDistBetweenStairsPoints_ && minHightDistBetweenPoints(debug_line_cloud) >= minHightDistBetweenAllStairsPoints_){
                    ROS_INFO("Staris; number points on line: %i", pointsAtLineCounter);

                    Eigen::Vector3f dir;
                    if(avg_point_per_cluster.at(best_pair_idx2)(2) >=base(2)){
                        dir=avg_point_per_cluster.at(best_pair_idx2)-base;
                    }else{
                        dir=base-avg_point_per_cluster.at(best_pair_idx2);
                    }
                    pcl::PointCloud<pcl::PointXYZ>::Ptr planeCloud(new pcl::PointCloud<pcl::PointXYZ>());
                    //is not needed while detecting stairs on unfilterd scancloud
//                                        stairsSreachPlaneDetection(input_surface_cloud, debug_line_cloud, base, dir, planeCloud);
                    publishResults(input_surface_cloud, planeCloud, clusters, final_cluster_idx, base, direction+base);
                }else{
                    ROS_INFO("No stairs, distance between points to large, or heightDistance between point too small");
                }

            }else{
                ROS_INFO("No stairs");
            }
//            sleep(2);
            iterationCounter=iterationCounter+1;
        }
    }
    ROS_INFO("No more possible stairs");

}

void HectorStairDetection::stairsSreachPlaneDetection(pcl::PointCloud<pcl::PointNormal>::Ptr &input_surface_cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr points_on_line, Eigen::Vector3f base, Eigen::Vector3f dir, pcl::PointCloud<pcl::PointXYZ>::Ptr &planeCloud){
    ROS_INFO("run plane segmentation");

    Eigen::Vector3f temp=dir.cross(Eigen::Vector3f(0,0,1));
    Eigen::Vector3f searchAxes=dir.cross(temp);
    searchAxes=searchAxes/searchAxes.squaredNorm();

    visualization_msgs::Marker line_list;
    line_list.header.frame_id=worldFrame_;
    line_list.id = 42;
    line_list.type = visualization_msgs::Marker::LINE_LIST;
    line_list.scale.x = 0.1;
    line_list.color.r = 1.0;
    line_list.color.a = 1.0;

    geometry_msgs::Point p;
    p.x = base(0);
    p.y = base(1);
    p.z = base(2);

    line_list.points.push_back(p);
    p.x = base(0) + searchAxes(0);
    p.y = base(1) + searchAxes(1);
    p.z = base(2) + searchAxes(2);
    line_list.points.push_back(p);

    line_marker_pub_.publish(line_list);

    pcl::PointCloud<pcl::PointXYZ>::Ptr searchCloud(new pcl::PointCloud<pcl::PointXYZ>());
    searchCloud->resize(input_surface_cloud->size());
    searchCloud->header.frame_id=worldFrame_;
    for (size_t i = 0; i < input_surface_cloud->points.size(); ++i)
    {
        const pcl::PointNormal &mls_pt = input_surface_cloud->points[i];
        pcl::PointXYZ pt(mls_pt.x, mls_pt.y, mls_pt.z);
        searchCloud->push_back(pt);
    }

    //search just in stairs environment
    Eigen::Vector2f xAxes;
    xAxes(0)=1;
    xAxes(1)=0;
    Eigen::Vector2f dir2f;
    dir2f(0)=dir(0);
    dir2f(1)=dir(1);
    float angleXToStairs=acos((dir2f.dot(xAxes))/(dir2f.norm()*xAxes.norm()));

    pcl::PointCloud<pcl::PointXYZ>::Ptr processCloud_v1(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr processCloud_v2(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(searchCloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.2, passThroughZMax_);
    pass.filter(*processCloud_v1);

    pass.setInputCloud(processCloud_v1);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(base(1)-fabs(sin(angleXToStairs)*maxDistBetweenStairsPoints_)-maxClusterXYDimension_/2, base(1)+fabs(sin(angleXToStairs)*maxDistBetweenStairsPoints_)+maxClusterXYDimension_/2);
    pass.filter(*processCloud_v2);

    pass.setInputCloud(processCloud_v2);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(base(0)-fabs(cos(angleXToStairs)*maxDistBetweenStairsPoints_)-maxClusterXYDimension_/2, base(0)+fabs(cos(angleXToStairs)*maxDistBetweenStairsPoints_)+maxClusterXYDimension_/2);
    pass.filter(*searchCloud);

    temp_after_pass_trough_pub_.publish(searchCloud);

    planeCloud->header.frame_id=worldFrame_;
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setModelType (pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (planeSegDistTresh_);
    seg.setAxis(searchAxes);
    seg.setEpsAngle(planeSegAngleEps_);

    seg.setInputCloud (searchCloud);
    seg.segment (*inliers, *coefficients);

    if (inliers->indices.size () == 0)
    {
        PCL_ERROR ("Could not estimate more planar models for the given dataset.");
    }

    ROS_DEBUG("extract plane and rest potins");
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (searchCloud);
    extract.setIndices (inliers);
    extract.setNegative(false);
    extract.filter (*planeCloud);
    planeCloud->header.frame_id=worldFrame_;

    //check if plane contains stairs points
    Eigen::Vector3f normal;
    normal(0)=coefficients->values[0];
    normal(1)=coefficients->values[1];
    normal(2)=coefficients->values[2];

    Eigen::Vector3f point;
    point(0)=0;
    point(1)=0;
    point(2)=-(coefficients->values[3]/coefficients->values[2]);

    Eigen::Vector3f normal_0;
    normal_0= normal/normal.squaredNorm();

    float d_hesse= point.dot(normal_0);

    bool isPossiblePlane=true;
    Eigen::Vector3f stairs_point;
    for(int i=0; i<points_on_line->size(); i++){
        stairs_point(0)=points_on_line->at(i).x;
        stairs_point(1)=points_on_line->at(i).y;
        stairs_point(2)=points_on_line->at(i).z;
        //        std::cout<<"hesse distance: "<<fabs(stairs_point.dot(normal_0)-d_hesse) <<std::endl;
        if(fabs(stairs_point.dot(normal_0)-d_hesse) > hesseTresh_){
            isPossiblePlane=false;
            break;
        }
    }

    if(isPossiblePlane){
        ROS_INFO("staris plane found");
        cloud_after_plane_detection_debug_pub_.publish(planeCloud);
    }else{
        planeCloud->resize(0);
    }
}

float HectorStairDetection::maxDistBetweenPoints(pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud){
    float max_dist=0;
    float temp_dist=0;
    for(int i=0; i<input_cloud->size(); i++){
        for(int j=i; j<input_cloud->size(); j++){
            temp_dist= std::sqrt(std::pow(input_cloud->at(i).x-input_cloud->at(j).x, 2)+std::pow(input_cloud->at(i).y-input_cloud->at(j).y, 2)+std::pow(input_cloud->at(i).z-input_cloud->at(j).z, 2));
            if(temp_dist>max_dist){
                max_dist=temp_dist;
            }
        }
    }
    return max_dist;
}

float HectorStairDetection::minHightDistBetweenPoints(pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud){
    float min_dist=FLT_MAX;
    float temp_dist=0;
    for(int i=0; i<input_cloud->size(); i++){
        for(int j=i+1; j<input_cloud->size(); j++){
            temp_dist= fabs(input_cloud->at(i).z-input_cloud->at(j).z);
            if(temp_dist<min_dist){
                min_dist=temp_dist;
            }
        }
    }
    return min_dist;
}

}


