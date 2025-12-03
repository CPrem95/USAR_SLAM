#ifndef MY_OPTIMIZE_HPP
#define MY_OPTIMIZE_HPP

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <random>

namespace ceres::examples {
    void publish_markers(rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_);
    double random_value(double min, double max);

    void publish_markers(rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_, 
        std::map<int, ceres::examples::Pose2d>& poses,
        const std::vector<Constraint2d>& constraints) 
        {
        auto marker_array = visualization_msgs::msg::MarkerArray();
        bool enable_node_text = true; // Flag to enable/disable text markers

        if (enable_node_text) {
            // Add text markers for each pose
            for (const auto& pair : poses) {
                visualization_msgs::msg::Marker text_marker;
                text_marker.header.frame_id = "map";
                text_marker.header.stamp = rclcpp::Clock().now();
                text_marker.ns = "example_text";
                text_marker.id = pair.first + 1000; // Offset to avoid ID conflicts with sphere markers
                text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
                text_marker.action = visualization_msgs::msg::Marker::ADD;

                // Set text position slightly above the sphere marker
                text_marker.pose.position.x = pair.second.x;
                text_marker.pose.position.y = pair.second.y;
                text_marker.pose.position.z = 0.5; // Offset above the sphere
                text_marker.pose.orientation.w = 1.0;

                // Set text content
                text_marker.text = " " + std::to_string(pair.first);

                // Set scale
                text_marker.scale.z = 0.2; // Text height

                // Set color (white)
                text_marker.color.r = 1.0;
                text_marker.color.g = 1.0;
                text_marker.color.b = 1.0;
                text_marker.color.a = 0.5;

                text_marker.lifetime = rclcpp::Duration::from_seconds(0); // Forever
                marker_array.markers.push_back(text_marker);
                }
        }

        // Create a SPHERE marker for each pose
        for (const auto& pair : poses) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = rclcpp::Clock().now();
            marker.ns = "example";
            marker.id = pair.first;
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;

            // Random positions
            marker.pose.position.x = pair.second.x;
            marker.pose.position.y = pair.second.y;
            marker.pose.position.z = 0.0;
            marker.pose.orientation.w = 1.0;

            // Set scale
            marker.scale.x = 0.25;
            marker.scale.y = 0.25;
            marker.scale.z = 0.25;

            if (pair.first == poses.rbegin()->first) {
            // Set scale for the last marker
            marker.scale.x = 0.4;
            marker.scale.y = 0.4;
            marker.scale.z = 0.4;
            // Set color (yellow) for the last marker
            marker.color.r = 1.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;
            } else {
            // Set scale for other markers
            marker.scale.x = 0.25;
            marker.scale.y = 0.25;
            marker.scale.z = 0.25;
            // Set color (red) for other markers
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;
            }

            marker.lifetime = rclcpp::Duration::from_seconds(0); // Forever
            marker_array.markers.push_back(marker);
        }
        
        // Create LINE_STRIP markers for each constraint
        int pose_edge_count = 0;
        int loop_edge_count = 20000; // Start from a high number to avoid duplicates with pose_edge_IDs
        
        for (const auto& constraint : constraints) {
            auto pose_begin_iter = poses.find(constraint.id_begin);
            auto pose_end_iter = poses.find(constraint.id_end);

            visualization_msgs::msg::Marker constraint_line;
            constraint_line.header.frame_id = "map";
            constraint_line.header.stamp = rclcpp::Clock().now();
            constraint_line.ns = "example_constraints";
            constraint_line.type = visualization_msgs::msg::Marker::LINE_STRIP;
            constraint_line.action = visualization_msgs::msg::Marker::ADD;

            // Set line points
            geometry_msgs::msg::Point p1, p2;
            if (pose_begin_iter != poses.end() && pose_end_iter != poses.end()) {
                p1.x = pose_begin_iter->second.x;
                p1.y = pose_begin_iter->second.y;
                p1.z = 0.0;
                p2.x = pose_end_iter->second.x;
                p2.y = pose_end_iter->second.y;
                p2.z = 0.0;
            }
            constraint_line.points.push_back(p1);
            constraint_line.points.push_back(p2);

            // Set line scale
            constraint_line.scale.x = 0.05;  // Line width
            
            if (constraint.id_end == constraint.id_begin + 1) { // Check if the constraint is between poses
                // Set line color (blue)
                constraint_line.color.r = 0.0;
                constraint_line.color.g = 0.0;
                constraint_line.color.b = 1.0;
                constraint_line.color.a = 0.8;
                constraint_line.id = pose_edge_count;  // Unique ID for consecutive constraints
                pose_edge_count++;
            } else { // Check if the constraint is loop closure
                // Set line color (green)
                constraint_line.color.r = 0.0;
                constraint_line.color.g = 1.0;
                constraint_line.color.b = 0.0;
                constraint_line.color.a = 1.0;
                constraint_line.id = loop_edge_count;  // Unique ID for non-consecutive constraints
                loop_edge_count++;
            }

            constraint_line.lifetime = rclcpp::Duration::from_seconds(0); // Forever
            marker_array.markers.push_back(constraint_line);
        }

        marker_publisher_->publish(marker_array);
        // RCLCPP_INFO(rclcpp::get_logger("ceres::examples"), "Published MarkerArray with lines");
    }

    double random_value(double min, double max) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(min, max);
        return dis(gen);
    }
}

#endif  // MY_OPTIMIZE_HPP
