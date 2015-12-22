# hector_stair_detection
# This message contains,<geometry_msgs::PoseStamped> representing the orientation of the stairs corrosponign to the border; and the direction as Vector

#Header information
std_msgs/Header header
#direction of staris (vector)
float32 directionX
float32 directionY
float32 directionZ
#orientation per staris
geometry_msgs/PoseStamped orientation_of_stairs
