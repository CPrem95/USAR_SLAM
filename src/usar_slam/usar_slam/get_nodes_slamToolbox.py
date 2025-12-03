import rclpy
from visualization_msgs.msg import MarkerArray

#!/usr/bin/env python3

import matplotlib.pyplot as plt
import scipy.io

def marker_callback(msg):
    node_ids = []
    xs = []
    ys = []
    nodes = []
    for marker in msg.markers:
        node_ids.append(marker.id)
        xs.append(marker.pose.position.x)
        ys.append(marker.pose.position.y)
        nodes.append([marker.id, marker.pose.position.x, marker.pose.position.y])
        print(f"Node ID: {marker.id}, X: {marker.pose.position.x}, Y: {marker.pose.position.y}")
    # print(f"Node IDs: {nodes}")
    scipy.io.savemat('slam_toolbox_nodes.mat', {'node_ids': node_ids, 'xs': xs, 'ys': ys, 'nodes': nodes})

    for i in range(len(nodes) - 1):
        if nodes[i][0] != nodes[i + 1][0] - 1:
            print(f"Duplicate node found: {nodes[i]}")
            
    plt.clf()
    plt.scatter(xs, ys, c='blue', label='Nodes')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('SLAM Toolbox Nodes')
    plt.legend()
    plt.pause(0.01)
    rclpy.shutdown()

def main():
    rclpy.init()
    node = rclpy.create_node('get_nodes_slam_toolbox')
    subscription = node.create_subscription(
        MarkerArray,
        '/slam_toolbox/graph_visualization',
        marker_callback,
        10
    )
    plt.ion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()