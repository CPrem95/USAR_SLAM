#include "include/my_optimize.hpp"

namespace ceres::examples {
    void publish_markers(rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_) {
        auto marker_array = visualization_msgs::msg::MarkerArray();

        for (int i = 0; i < 5; ++i) {  // Create 5 markers
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = this->get_clock()->now();
            marker.ns = "example";
            marker.id = i;
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;

            // Random positions
            marker.pose.position.x = random_value(-5, 5);
            marker.pose.position.y = random_value(-5, 5);
            marker.pose.position.z = 0.5;
            marker.pose.orientation.w = 1.0;

            // Set scale
            marker.scale.x = 0.5;
            marker.scale.y = 0.5;
            marker.scale.z = 0.5;

            // Set color (red)
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;

            marker.lifetime = rclcpp::Duration::from_seconds(0); // Forever
            marker_array.markers.push_back(marker);
        }

        publisher_->publish(marker_array);
        RCLCPP_INFO(this->get_logger(), "Published MarkerArray");
    }

    double random_value(double min, double max) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(min, max);
        return dis(gen);
    }
} // namespace ceres::examples