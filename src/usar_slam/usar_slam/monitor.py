#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String   # Example message type
from sensor_msgs.msg import Image # Example message type
from std_msgs.msg import Float32MultiArray
import time
from nav_msgs.msg import Odometry


class FrequencyMonitor(Node):
    def __init__(self):
        super().__init__('frequency_monitor')

        # Dictionary to store last timestamps and frequencies
        self.topic_stats = {
            '/UWBradar0/readings': {'last_time': None, 'freq': 0.0},
            '/UWBradar1/readings': {'last_time': None, 'freq': 0.0},
            '/UWBradar2/readings': {'last_time': None, 'freq': 0.0},
            '/UWBradar3/readings': {'last_time': None, 'freq': 0.0},
            '/UWBradar4/readings': {'last_time': None, 'freq': 0.0},
            '/UWBradar5/readings': {'last_time': None, 'freq': 0.0},
            '/odom': {'last_time': None, 'freq': 0.0},
        }

        # Subscribers
        self.create_subscription(
            Float32MultiArray,
            '/UWBradar0/readings',
            self.chatter_callback,
            10
        )
        self.create_subscription(
            Float32MultiArray,
            '/UWBradar1/readings',
            self.chatter_callback,
            10
        )
        self.create_subscription(
            Float32MultiArray,
            '/UWBradar2/readings',
            self.chatter_callback,
            10
        )
        self.create_subscription(
            Float32MultiArray,
            '/UWBradar3/readings',
            self.chatter_callback,
            10
        )
        self.create_subscription(
            Float32MultiArray,
            '/UWBradar4/readings',
            self.chatter_callback,
            10
        )
        self.create_subscription(
            Float32MultiArray,
            '/UWBradar5/readings',
            self.chatter_callback,
            10
        )
        self.create_subscription(
            Odometry,
            '/odom',
            self.chatter_callback,
            10
        )

        # Timer to print statistics every 5 seconds
        self.timer = self.create_timer(5.0, self.print_stats)

    def _update_frequency(self, topic_name: str):
        """Update frequency for given topic using timestamp differences."""
        now = time.time()
        last_time = self.topic_stats[topic_name]['last_time']

        if last_time is not None:
            dt = now - last_time
            if dt > 0:
                self.topic_stats[topic_name]['freq'] = 1.0 / dt

        self.topic_stats[topic_name]['last_time'] = now

    def chatter_callback(self, msg: String):
        self._update_frequency('/chatter')

    def image_callback(self, msg: Image):
        self._update_frequency('/camera/image_raw')

    def print_stats(self):
        self.get_logger().info("------ Topic Frequencies ------")
        for topic, stats in self.topic_stats.items():
            freq = stats['freq']
            self.get_logger().info(f"{topic}: {freq:.2f} Hz")
        self.get_logger().info("-------------------------------")

def main(args=None):
    rclpy.init(args=args)
    node = FrequencyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
