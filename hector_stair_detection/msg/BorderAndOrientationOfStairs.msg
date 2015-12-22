# hector_stair_detection
# This message contains, <visualization_msgs::MarkerArray> representing the border of the staris, and <geometry_msgs::PoseStamped> representing the orientation of the stairs corrosponign to the border

#Header information
std_msgs/Header header
#border per staris
visualization_msgs/MarkerArray border_of_stairs
#orientation per staris
geometry_msgs/PoseStamped orientation_of_stairs
#number of points on line; more points <=> larger stairs <=> higher accurancy
int32 number_of_points
#direction of staris (vector)
float32 directionX
float32 directionY
float32 directionZ
