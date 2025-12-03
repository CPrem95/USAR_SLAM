# USAR_SLAM  

Clone the repository (i.e. workspace).
Go to the downloaded workspace and build it.  
```
cd USAR_SLAM
```
```
colcon build
```
Find the datasets (i.e. ROSBAG files over here):
```
Link will be placed after anonymising is over!
```
Now you have the complete USAR SLAM system and the datasets.  
You can recreate the results in the video now.  
First, open a **new terminal**, and run the CERES graph SLAM backend.  
```
ros2 run my_ceres opti_node 
```
Open a **new terminal**, and run the USAR_SLAM backend.  
```
ros2 run usar_slam node
```
Open a **new terminal**, go to the downloaded rosbags, and play the rosbag file.
```
ros2 bag play rosbag2_2025_08_27-18_25_42_0.db3
```
### Visualization
Open RViz in a new terminal  
```
rviz2
```
Go to **Add** &larr; **By topic** &larr; **/visualization_marker_array/MarkerArray** to visualize the pose graph within rviz2.  

  
You can see the results now!  
Meanwhile, read the following instructions carefully.  

Once you have finished teleoperation (i.e. the bag file stopped playing), run the following command to stop SLAM and to deplete the accumulated readings.  
**New terminal**  
```
ros2 topic pub /stop_slam std_msgs/msg/Bool '{data: true}' --once 
```
Finally, run the following command to optimize the complete pose graph.  
```
ros2 topic pub /optimize std_msgs/msg/Bool '{data: true}' --once 
```
### Developer NOTES:  
The sensor configuration corresponds to the Fig. 3 in the paper.  
The sensor IDs (UWB Radar: Left and Right) also follow the same.  
TurtleBot's default bringup topics are their default (e.g. /cmd_vel, /odom).  
UWB Radar readings [raw]:  
Left:  `/UWBradar1/readings`  
Right: `/UWBradar2/readings`  
