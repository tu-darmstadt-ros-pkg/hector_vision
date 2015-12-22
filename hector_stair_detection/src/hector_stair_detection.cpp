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

    possible_stairs_cloud_pub_= nh.advertise<pcl::PointCloud<pcl::PointXYZI> >("/hector_stair_detection/possible_stairs_cloud", 100, true);
    points_on_line_cloud_debug_= nh.advertise<pcl::PointCloud<pcl::PointXYZI> >("/hector_stair_detection/point_on_line_debug", 100, true);
    surfaceCloud_pub_debug_= nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("/hector_stair_detection/surfaceCloud_debug", 100, true);
    final_stairs_cloud_pub_=  nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("/hector_stair_detection/final_stairs_cloud", 100, true);
    border_of_stairs_pub_= nh.advertise<visualization_msgs::MarkerArray>("/hector_stair_detection/boarder_of_stairs", 100, true);
    stairs_position_and_orientaion_pub_= nh.advertise<geometry_msgs::PoseStamped>("/hector_stair_detection/stairs_orientation", 100, true);
    stairs_position_and_orientaion_with_direction_pub_= nh.advertise<hector_stair_detection::PositionAndOrientaion>("/hector_stair_detection/stairs_orientaion_as_vector", 100, true);
    border_and_orientation_stairs_combined_pub_= nh.advertise<hector_stair_detection::BorderAndOrientationOfStairs>("/hector_stair_detection/border_and_orientation_of_stairs", 100, true);

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
        pcl_sub = nh.subscribe("/openni/depth/points", 1, &HectorStairDetection::PclCallback, this);
        //        pcl_sub = nh.subscribe("/hector_octomap_server/octomap_point_cloud_centers", 1, &HectorStairDetection::PclCallback, this);
    }

}

HectorStairDetection::~HectorStairDetection()
{}

void HectorStairDetection::publishResults(pcl::PointCloud<pcl::PointNormal>::Ptr &input_surface_cloud,
                                          pcl::IndicesClustersPtr cluster_indices, std::vector<int> final_cluster_idx, Eigen::Vector3f base, Eigen::Vector3f point){

    geometry_msgs::PoseStamped position_and_orientaion;
    Eigen::Vector3f directionStairs;
    getStairsPositionAndOrientation(base, point, input_surface_cloud->header.frame_id, directionStairs, position_and_orientaion);
    hector_stair_detection::PositionAndOrientaion pos_and_orientaion_message;
    pos_and_orientaion_message.orientation_of_stairs=position_and_orientaion;
    pos_and_orientaion_message.directionX=directionStairs(0);
    pos_and_orientaion_message.directionY=directionStairs(1);
    pos_and_orientaion_message.directionZ=directionStairs(2);

    //compute and pubish max and min boarder of the stairs
    pcl::PointCloud<pcl::PointXYZ>::Ptr final_stairsCloud(new pcl::PointCloud<pcl::PointXYZ>);
    visualization_msgs::MarkerArray stairs_boarder_marker;
    getFinalStairsCloud_and_position(input_surface_cloud->header.frame_id, directionStairs, input_surface_cloud, cluster_indices, final_cluster_idx, final_stairsCloud, stairs_boarder_marker);

    if(final_stairs_cloud_pub_.getNumSubscribers()>0){
        final_stairs_cloud_pub_.publish(final_stairsCloud);
    }
    if(border_of_stairs_pub_.getNumSubscribers()>0){
        border_of_stairs_pub_.publish(stairs_boarder_marker);
    }

    hector_stair_detection::BorderAndOrientationOfStairs border_and_orientation_msg;
    border_and_orientation_msg.header.frame_id=input_surface_cloud->header.frame_id;
    border_and_orientation_msg.border_of_stairs= stairs_boarder_marker;
    border_and_orientation_msg.orientation_of_stairs=position_and_orientaion;
    border_and_orientation_msg.number_of_points= final_cluster_idx.size();
    border_and_orientation_msg.directionX=directionStairs(0);
    border_and_orientation_msg.directionY=directionStairs(1);
    border_and_orientation_msg.directionZ=directionStairs(2);

    //refine orientaion (not working)
    //    Eigen::Vector2f directionStairs2D;
    //    directionStairs2D(0)=directionStairs(0);
    //    directionStairs2D(1)=directionStairs(1);
    //    Eigen::Vector2f minXminY;
    //    minXminY(0)=stairs_boarder_marker.markers.at(0).pose.position.x;
    //    minXminY(1)=stairs_boarder_marker.markers.at(0).pose.position.y;
    //    Eigen::Vector2f maxXminY;
    //    maxXminY(0)=stairs_boarder_marker.markers.at(3).pose.position.x;
    //    maxXminY(1)=stairs_boarder_marker.markers.at(0).pose.position.y;
    //    Eigen::Vector2f minXmaxY;
    //    minXmaxY(0)=stairs_boarder_marker.markers.at(0).pose.position.x;
    //    minXmaxY(1)=stairs_boarder_marker.markers.at(3).pose.position.y;
    //    refineOrientaion(directionStairs2D, minXminY, maxXminY, minXmaxY, position_and_orientaion);

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

void HectorStairDetection::getStairsPositionAndOrientation(Eigen::Vector3f base, Eigen::Vector3f point, std::string frameID, Eigen::Vector3f &direction, geometry_msgs::PoseStamped &position_and_orientaion){
    if(point(2) >=base(2)){
        direction=point-base;
    }else{
        direction=base-point;
    }

    Eigen::Vector3f stairs_position= 0.5*(base + point);

    position_and_orientaion.header.frame_id=frameID;
    position_and_orientaion.pose.position.x=stairs_position(0);
    position_and_orientaion.pose.position.y=stairs_position(1);
    position_and_orientaion.pose.position.z=stairs_position(2);
    float stairs_yaw= atan2(direction(1), direction(0));
    float staris_pitch= atan2(direction(1)*sin(stairs_yaw)+direction(0)*cos(stairs_yaw), direction(2))+M_PI_2;
    tf::Quaternion temp;
    temp.setEulerZYX(stairs_yaw,staris_pitch,0.0);
    position_and_orientaion.pose.orientation.x=temp.getX();
    position_and_orientaion.pose.orientation.y=temp.getY();
    position_and_orientaion.pose.orientation.z=temp.getZ();
    position_and_orientaion.pose.orientation.w=temp.getW();
}

void HectorStairDetection::getFinalStairsCloud_and_position(std::string frameID, Eigen::Vector3f directionS, pcl::PointCloud<pcl::PointNormal>::Ptr &input_surface_cloud, pcl::IndicesClustersPtr cluster_indices, std::vector<int> final_cluster_idx,
                                                            pcl::PointCloud<pcl::PointXYZ>::Ptr &final_stairsCloud, visualization_msgs::MarkerArray &stairs_boarder_marker){
    int cluster_counter=0;
    float minX= FLT_MAX;
    float maxX= -FLT_MAX;
    float minY= FLT_MAX;
    float maxY= -FLT_MAX;
    float minZ= FLT_MAX;
    float maxZ= -FLT_MAX;
    final_stairsCloud->header.frame_id=frameID;

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

                if(tempP.x < minX){
                    minX=tempP.x;
                }

                if(tempP.x > maxX){
                    maxX=tempP.x;
                }

                if(tempP.y < minY){
                    minY=tempP.y;
                }

                if(tempP.y > maxY){
                    maxY=tempP.y;
                }

                if(tempP.z < minZ){
                    minZ=tempP.z;
                }

                if(tempP.z > maxZ){
                    maxZ=tempP.z;
                }


            }
        }
        cluster_counter=cluster_counter+1;
    }

    //TODO::
    //    //project end of stairs to the ground
    //    if(minZ>0.1){
    //        getEndOfStairs
    //    }

    Eigen::Vector2f directionStairs;
    directionStairs(0)=directionS(0);
    directionStairs(1)=directionS(1);
    Eigen::Vector2f minXminY;
    minXminY(0)=minX;
    minXminY(1)=minY;
    Eigen::Vector2f maxXminY;
    maxXminY(0)=maxX;
    maxXminY(1)=minY;
    Eigen::Vector2f minXmaxY;
    minXmaxY(0)=minX;
    minXmaxY(1)=maxY;
    int componetOfDirection= getZComponent(directionStairs, minXminY, maxXminY, minXmaxY);

    for(int i=0; i<4; i++){
        visualization_msgs::Marker marker;
        marker.header.frame_id = frameID;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.ns = "hector_stair_detection";
        marker.id = i;
        if(i==0){
            marker.pose.position.x = minX;
            marker.pose.position.y = minY;
            switch(componetOfDirection) {
            case 1: marker.pose.position.z=maxZ;
                std::cout<<"1"<<std::endl;
                break;
            case 2: marker.pose.position.z=maxZ;
                std::cout<<"2"<<std::endl;
                break;
            case 3: marker.pose.position.z=minZ;
                std::cout<<"3"<<std::endl;
                break;
            case 4: marker.pose.position.z=minZ;
                std::cout<<"4"<<std::endl;
                break;

            }
        }
        if(i==1){
            marker.pose.position.x = minX;
            marker.pose.position.y = maxY;
            switch(componetOfDirection) {
            case 1: marker.pose.position.z=maxZ;
                break;
            case 2: marker.pose.position.z=minZ;
                break;
            case 3: marker.pose.position.z=minZ;
                break;
            case 4: marker.pose.position.z=maxZ;
                break;

            }
        }
        if(i==2){
            marker.pose.position.x = maxX;
            marker.pose.position.y = minY;
            switch(componetOfDirection) {
            case 1: marker.pose.position.z=minZ;
                break;
            case 2: marker.pose.position.z=maxZ;
                break;
            case 3: marker.pose.position.z=maxZ;
                break;
            case 4: marker.pose.position.z=minZ;
                break;

            }
        }
        if(i==3){
            marker.pose.position.x = maxX;
            marker.pose.position.y = maxY;
            switch(componetOfDirection) {
            case 1: marker.pose.position.z=minZ;
                break;
            case 2: marker.pose.position.z=minZ;
                break;
            case 3: marker.pose.position.z=maxZ;
                break;
            case 4: marker.pose.position.z=maxZ;
                break;

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

void HectorStairDetection::refineOrientaion(Eigen::Vector2f directionStairs, Eigen::Vector2f minXminY, Eigen::Vector2f maxXminY, Eigen::Vector2f minXmaxY, geometry_msgs::PoseStamped &position_and_orientaion){
    Eigen::Vector2f direction1;
    Eigen::Vector2f direction2;

    direction1=maxXminY-minXminY;
    direction2=minXmaxY-minXminY;

    float angle1=acos((directionStairs.dot(direction1))/(directionStairs.norm()*direction1.norm()));
    float angle2=acos((directionStairs.dot(direction2))/(directionStairs.norm()*direction2.norm()));

    angle1= angle1 -(M_PI*floor(angle1/M_PI));
    angle2= angle2 -(M_PI*floor(angle2/M_PI));

    double refined_yaw;
    if(angle1>angle2){
        refined_yaw=atan2(direction1(1), direction1(0));
    }else{
        refined_yaw=atan2(direction2(1), direction2(0));
    }

    tf::Quaternion q_tf;
    tf::quaternionMsgToTF(position_and_orientaion.pose.orientation, q_tf);

    double r, p, y;
    tf::Matrix3x3(q_tf).getEulerZYX(y,p,r);
    tf::Quaternion temp;
    temp.setEulerZYX(refined_yaw,p,0.0);
    position_and_orientaion.pose.orientation.x=temp.getX();
    position_and_orientaion.pose.orientation.y=temp.getY();
    position_and_orientaion.pose.orientation.z=temp.getZ();
    position_and_orientaion.pose.orientation.w=temp.getW();
}

void HectorStairDetection::getPreprocessedCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud, pcl::PointCloud<pcl::PointNormal>::Ptr &output_cloud){
    ROS_INFO("Hector Stair Detection get Surface");
    pcl::PointCloud<pcl::PointXYZ>::Ptr processCloud_v1(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr processCloud_v2(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(input_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(passThroughZMin_, passThroughZMax_);
    pass.filter(*processCloud_v1);

    pcl::VoxelGrid<pcl::PointXYZ> vox;
    vox.setInputCloud(processCloud_v1);
    vox.setLeafSize(voxelGridX_, voxelGridY_, voxelGridZ_);
    vox.setDownsampleAllData(false);
    vox.filter(*processCloud_v2);

    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;

    mls.setSearchRadius(0.05);
    mls.setPolynomialOrder(1);
    mls.setComputeNormals(true);
    mls.setInputCloud(processCloud_v2);

    boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > point_cloud_mls_normal;
    point_cloud_mls_normal.reset(new pcl::PointCloud<pcl::PointNormal>);
    mls.process(*point_cloud_mls_normal);

    float x;
    float y;
    float z;
    float n_x;
    float n_y;
    float n_z;

    output_cloud->clear();
    output_cloud->header.frame_id=input_cloud->header.frame_id;
    if(!refineSurfaceRequired_){
        output_cloud.reset(new pcl::PointCloud<pcl::PointNormal>(*point_cloud_mls_normal));
    }else{
        bool  pushback=true;
        for(int i=0; i<point_cloud_mls_normal->size(); i++){
            pushback=true;
            x = point_cloud_mls_normal->at(i).x;
            y = point_cloud_mls_normal->at(i).y;
            z = point_cloud_mls_normal->at(i).z;
            n_x = point_cloud_mls_normal->at(i).normal_x;
            n_y = point_cloud_mls_normal->at(i).normal_y;
            n_z = point_cloud_mls_normal->at(i).normal_z;
            std::vector<int> eraseIdx;
            for(int j=0; j<output_cloud->size(); j++){
                if(fabs(output_cloud->at(j).x - x) < 0.01 && fabs(output_cloud->at(j).y - y)<0.01){
                    if(z > output_cloud->at(j).z){
                        eraseIdx.push_back(j);
                    }else{
                        pushback=false;
                    }
                }
            }

            for(int c=0; c<eraseIdx.size(); c++){
                output_cloud->erase(output_cloud->begin()+eraseIdx.at(c));
            }

            if(pushback){
                pcl::PointNormal pushbackN;
                pushbackN.x=x;
                pushbackN.y=y;
                pushbackN.z=z;
                pushbackN.normal_x=n_x;
                pushbackN.normal_y=n_y;
                pushbackN.normal_z=n_z;
                output_cloud->push_back(pushbackN);
            }
        }
    }

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

    if(fabs(d1.dot(point_b_normal)) < 0.5 && fabs(d2.dot(point_b_normal)) < 0.5){
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
            //TODO:: remove
            points_on_line_cloud_debug_.publish(debug_line_cloud);
            if(pointsAtLineCounter >= minCountPointsAtLine){
                //publish results
                if(points_on_line_cloud_debug_.getNumSubscribers()>0){
                    points_on_line_cloud_debug_.publish(debug_line_cloud);
                }
                if(maxDistBetweenPoints(debug_line_cloud) <= maxDistBetweenStairsPoints_ || minHightDistBetweenPoints(debug_line_cloud) > minHightDistBetweenAllStairsPoints_){
                    ROS_INFO("Staris; number points on line: %i", pointsAtLineCounter);
                    publishResults(input_surface_cloud, clusters, final_cluster_idx, base, direction+base);
                }else{
                    ROS_INFO("No stairs, distance between points to large, or heightDistance between point too small");
                }

            }else{
                ROS_INFO("No stairs");
            }
        }else{
            ROS_INFO("No stairs");
        }
    }else{
        ROS_INFO("No stairs");
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


