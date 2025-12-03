# USAR_SLAM

Once you have finished teleoperation (i.e. the bag file stopped playing), run the following command to stop SLAM and to deplete the accumulated readings.  
```
ros2 topic pub /stop_slam std_msgs/msg/Bool '{data: true}' --once 
```
Finally, run the following command to optimize the complete pose graph.  
```
ros2 topic pub /optimize std_msgs/msg/Bool '{data: true}' --once 
```


