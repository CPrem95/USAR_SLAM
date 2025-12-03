#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "angle_manifold.h"
#include "ceres/ceres.h"
#include "my_read_g2o.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "pose_graph_2d_error_term.h"
#include "types.h"
#include "my_optimize.hpp"

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "std_msgs/msg/bool.hpp"

namespace ceres::examples {
namespace {

// Constructs the nonlinear least squares optimization problem from the pos`e
// graph constraints.
void BuildOptimizationProblem(const std::vector<Constraint2d>& constraints,
                              std::map<int, Pose2d>* poses,
                              ceres::Problem* problem) {
  CHECK(poses != nullptr);
  CHECK(problem != nullptr);
  if (constraints.empty()) {
    LOG(INFO) << "No constraints, no problem to optimize.";
    return;
  }

  ceres::LossFunction* loss_function = nullptr;
  ceres::Manifold* angle_manifold = AngleManifold::Create();

  for (const auto& constraint : constraints) {
    auto pose_begin_iter = poses->find(constraint.id_begin);
    CHECK(pose_begin_iter != poses->end())
        << "Pose with ID: " << constraint.id_begin << " not found.";
    auto pose_end_iter = poses->find(constraint.id_end);
    CHECK(pose_end_iter != poses->end())
        << "Pose with ID: " << constraint.id_end << " not found.";

    const Eigen::Matrix3d sqrt_information =
        constraint.information.llt().matrixL();
    // Ceres will take ownership of the pointer.
    ceres::CostFunction* cost_function = PoseGraph2dErrorTerm::Create(
        constraint.x, constraint.y, constraint.yaw_radians, sqrt_information);
    problem->AddResidualBlock(cost_function,
                              loss_function,
                              &pose_begin_iter->second.x,
                              &pose_begin_iter->second.y,
                              &pose_begin_iter->second.yaw_radians,
                              &pose_end_iter->second.x,
                              &pose_end_iter->second.y,
                              &pose_end_iter->second.yaw_radians);

    problem->SetManifold(&pose_begin_iter->second.yaw_radians, angle_manifold);
    problem->SetManifold(&pose_end_iter->second.yaw_radians, angle_manifold);
  }

  // The pose graph optimization problem has three DOFs that are not fully
  // constrained. This is typically referred to as gauge freedom. You can apply
  // a rigid body transformation to all the nodes and the optimization problem
  // will still have the exact same cost. The Levenberg-Marquardt algorithm has
  // internal damping which mitigate this issue, but it is better to properly
  // constrain the gauge freedom. This can be done by setting one of the poses
  // as constant so the optimizer cannot change it.
  auto pose_start_iter = poses->begin();
  CHECK(pose_start_iter != poses->end()) << "There are no poses.";
  problem->SetParameterBlockConstant(&pose_start_iter->second.x);
  problem->SetParameterBlockConstant(&pose_start_iter->second.y);
  problem->SetParameterBlockConstant(&pose_start_iter->second.yaw_radians);
}

// Returns true if the solve was successful.
bool SolveOptimizationProblem(ceres::Problem* problem) {
  CHECK(problem != nullptr);

  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);

  std::cout << summary.FullReport() << '\n';

  return summary.IsSolutionUsable();
}

// Output the poses to the file with format: ID x y yaw_radians.
bool OutputPoses(const std::string& filename,
                 const std::map<int, Pose2d>& poses) {
  std::fstream outfile;
  outfile.open(filename.c_str(), std::istream::out);
  if (!outfile) {
    std::cerr << "Error opening the file: " << filename << '\n';
    return false;
  }
  for (const auto& pair : poses) {
    outfile << pair.first << " " << pair.second.x << " " << pair.second.y << ' '
            << pair.second.yaw_radians << '\n';
  }

  return true;
}

std_msgs::msg::Float32MultiArray PosesToMsg(const std::map<int, Pose2d>& poses) {
  std_msgs::msg::Float32MultiArray msg;
  // Each pose: [id, x, y, yaw]
  msg.data.reserve(poses.size() * 4);
  for (const auto& pair : poses) {
    msg.data.push_back(static_cast<float>(pair.first));
    msg.data.push_back(static_cast<float>(pair.second.x));
    msg.data.push_back(static_cast<float>(pair.second.y));
    msg.data.push_back(static_cast<float>(pair.second.yaw_radians));
  }
  return msg;
}



}  // namespace
}  // namespace ceres::examples

class CeresSubscriber : public rclcpp::Node
{
public:
  CeresSubscriber(int argc, char** argv, std::map<int, ceres::examples::Pose2d>& poses, std::vector<ceres::examples::Constraint2d>& constraints, ceres::Problem& problem) : Node("ceres_subscriber"), 
          argc_(argc), argv_(argv)
  {
    pose_subscription_ = this->create_subscription<geometry_msgs::msg::Pose2D>(
      "pose", 10,
      [this](const geometry_msgs::msg::Pose2D::SharedPtr msg) {
        this->pose_callback(msg);
      });
    
    edge_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
      "edge", 10,
      [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        this->edge_callback(msg);
      });

    loop_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
      "loop", 10,
      [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        this->loop_callback(msg);
      });

    optimize_subscription_ = this->create_subscription<std_msgs::msg::Bool>(
      "optimize", 10,
      [this](const std_msgs::msg::Bool::SharedPtr msg) {
        this->optimize_callback(msg);
      });

    marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("visualization_marker_array", 10);
    optimized_flag_publisher_ = this->create_publisher<std_msgs::msg::Bool>("optimized_flag", 10);
    optimized_poses_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("optimized_poses", 10);

    accumulation_thread_ = std::thread([this, &problem, &poses, &constraints]() {
      rclcpp::Rate rate(10); // 10 Hz
      RCLCPP_INFO(this->get_logger(), "\033[1;31mInitialization completed. Waiting for data...\033[0m");
      while (rclcpp::ok()) {
        {
          std::unique_lock<std::mutex> lock(global_mutex_);
          if (pose_data_ready_) {
            auto [pose_x, pose_y, pose_th] = pose_data_; // Extract pose_data
            pose_data_ready_ = false;
            RCLCPP_INFO(this->get_logger(), "\033[1;33mProcessing Pose ID: %d\033[0m", pose_id);
            RCLCPP_INFO(this->get_logger(), "Added Pose data: x=%.2f, y=%.2f, theta=%.2f",
              pose_x, pose_y, pose_th);
            ceres::examples::ReadVertex2(pose_id, pose_data_, &poses);
            lock.unlock();
          }
        }

        {
          std::unique_lock<std::mutex> lock(global_mutex_);
          if (edge_data_ready_) {
            auto [edge_dx, edge_dy, edge_dth] = edge_data_; // Extract edge_data
            edge_data_ready_ = false;
            RCLCPP_INFO(this->get_logger(), "Added Edge data: dx=%.2f, dy=%.2f, dth=%.2f",
              edge_dx, edge_dy, edge_dth);
            ceres::examples::ReadConstraint2(edge_id_ini, edge_id_end, edge_data_, edge_information_, &constraints);
            lock.unlock();
          }
        }

        {
          std::unique_lock<std::mutex> lock(global_mutex_);
          if (loop_data_ready_) {
            // auto [loop_dx, loop_dy, loop_dth] = loop_data_; // Extract loop_data
            loop_data_ready_ = false;
            RCLCPP_INFO(this->get_logger(), "\033[1;34mAdded Loop closure: id_ini=%d, id_end=%d\033[0m", loop_id_ini, loop_id_end);
            ceres::examples::ReadConstraint2(loop_id_ini, loop_id_end, loop_data_, loop_information_, &constraints);
            loop_id++; // Increment loop count
            lock.unlock();
          }
        }
        
        // if (loop_count == 10) {
        //     RCLCPP_INFO(this->get_logger(), "\033[1;31mNumber of poses: %zu\033[0m", poses.size());
        //     RCLCPP_INFO(this->get_logger(), "\033[1;31mNumber of constraints: %zu\033[0m", constraints.size());
        // }
      }
    });

    optimize_thread_ = std::thread([this, &problem, &poses, &constraints]() {
      int previous_pose_id = 0;
      rclcpp::Rate rate(10); // 10 Hz
      auto optimized_msg = std_msgs::msg::Bool();
      optimized_msg.data = true;
      while (rclcpp::ok()) {
      if ((pose_id % 500 == 0 && pose_id != 0 && pose_id != previous_pose_id) || pose_id == 3499 || external_optimize) {
        external_optimize = false; // Reset the flag
        RCLCPP_INFO(this->get_logger(), "\033[1;31mOptimizing the pose graph...\033[0m");
        std::unique_lock<std::mutex> lock(global_mutex_);
        ceres::examples::BuildOptimizationProblem(constraints, &poses, &problem);
        lock.unlock();
        CHECK(ceres::examples::OutputPoses("poses_original.txt", poses))
        << "Error outputting to poses_original.txt";
        CHECK(ceres::examples::SolveOptimizationProblem(&problem))
        << "The solve was not successful, exiting.";
        CHECK(ceres::examples::OutputPoses("poses_optimized.txt", poses))
        << "Error outputting to poses_original.txt";
        // lock.unlock();
        RCLCPP_INFO(this->get_logger(), "\033[1;32mPose graph optimization completed.\033[0m");
        
        if (previous_pose_id == 3499) {
          std::this_thread::sleep_for(std::chrono::seconds(2));
          RCLCPP_INFO(this->get_logger(), "\033[1;31mOptimization completed. Shutting down...\033[0m");
          rclcpp::shutdown();
        }
        previous_pose_id = pose_id; // Update the previous pose count

        optimized_flag_publisher_->publish(optimized_msg); // Publish optimization status
        optimized_poses_publisher_->publish(ceres::examples::PosesToMsg(poses)); // Publish optimized poses
      }
      rate.sleep();
      }
    });

    visualization_thread_ = std::thread([this, &poses, &constraints]() {
      rclcpp::Rate rate(10); // 10 Hz
      while (rclcpp::ok()) {
        ceres::examples::publish_markers(marker_publisher_, poses, constraints);
        rate.sleep();
      }
    });

    rclcpp::on_shutdown([this]() {
      if (accumulation_thread_.joinable()) {
        accumulation_thread_.join();
      }
      if (optimize_thread_.joinable()) {
        optimize_thread_.join();
      }
    });
  }

private:
  rclcpp::Subscription<geometry_msgs::msg::Pose2D>::SharedPtr pose_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr optimize_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr edge_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr loop_subscription_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr optimized_flag_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr optimized_poses_publisher_;

  std::thread accumulation_thread_;
  std::thread optimize_thread_;
  std::thread visualization_thread_;

  int argc_;
  char** argv_;
  std::tuple<double, double, double> pose_data_; // Tuple to store pose data
  std::tuple<double, double, double> edge_data_; // Tuple to store edge data
  std::tuple<double, double, double> loop_data_; // Tuple to store loop data
  std::tuple<double, double, double, double, double, double> edge_information_; // Tuple to store edge information
  std::tuple<double, double, double, double, double, double> loop_information_; // Tuple to store loop information
  
  std::mutex pose_mutex_; // Mutex for thread synchronization
  std::mutex edge_mutex_; // Mutex for thread synchronization
  std::mutex loop_mutex_; // Mutex for thread synchronization
  std::mutex global_mutex_; // Mutex for thread synchronization

  std::condition_variable pose_data_condition_; // Condition variable for signaling
  std::condition_variable edge_data_condition_; // Condition variable for signaling
  std::condition_variable loop_data_condition_; // Condition variable for signaling
  bool pose_data_ready_ = false; // Flag to indicate data readiness
  bool edge_data_ready_ = false; // Flag to indicate data readiness
  bool loop_data_ready_ = false; // Flag to indicate data readiness

  int pose_id = -1; // Variable to store vertex id
  int edge_id = -1;
  int loop_id = -1;

  double pose_x = 0.0; // Variable to store msg->x
  double pose_y = 0.0; // Variable to store msg->y
  double pose_th = 0.0; // Variable to store msg->theta

  int edge_id_ini = 0; // Variable to store initial vertex id
  int edge_id_end = 0; // Variable to store final vertex id
  double edge_dx = 0.0; // Variable to store dx
  double edge_dy = 0.0; // Variable to store dy
  double edge_dth = 0.0; // Variable to store dtheta

  int loop_id_ini = 0; // Variable to store initial vertex id
  int loop_id_end = 0; // Variable to store final vertex id
  double loop_dx = 0.0; // Variable to store dx
  double loop_dy = 0.0; // Variable to store dy
  double loop_dth = 0.0; // Variable to store dtheta

  bool text_write = true; // Flag to indicate if text write is enabled
  // std::istringstream input_stream("0 0 0.0 0.0 0.0"); // Example input stream
  // input_stream >> loop_id_ini >> loop_id_end >> loop_dx >> loop_dy >> loop_dth; // Example input stream usage

  bool external_optimize = false; // Flag to notify external optimization

  void pose_callback(const geometry_msgs::msg::Pose2D::SharedPtr msg)
  {
    pose_x = msg->x; // Store msg->x in the class variable
    pose_y = msg->y; // Store msg->y in the class variable
    pose_th = msg->theta; // Store msg->theta in the class variable
    // Send the received pose data to the processing thread
    std::lock_guard<std::mutex> lock(global_mutex_);
    pose_data_ = {pose_x, pose_y, pose_th};
    pose_id++;
    pose_data_ready_ = true;
    pose_data_condition_.notify_one();
    
    if (text_write) { // Check if text_write is enabled
      // Append pose_data to a txt file
      std::ofstream pose_file("my_pose_data.txt", std::ios::app);
      if (pose_file.is_open()) {
        pose_file << "VERTEX_SE2" << " " << pose_id << " " << pose_x << " " << pose_y << " " << pose_th << "\n";
        pose_file.close();
        RCLCPP_INFO(this->get_logger(), "Pose data appended to pose_data.txt");
      } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to open pose_data.txt for appending");
      }
    }
  }

  void edge_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
  {
    edge_id_ini = msg->data[0]; // Store initial vertex id in the class variable
    edge_id_end = msg->data[1]; // Store final vertex id in the class variable
    edge_dx = msg->data[2]; // Store dx in the class variable
    edge_dy = msg->data[3]; // Store dy in the class variable
    edge_dth = msg->data[4]; // Store dtheta in the class variable
    
    // RCLCPP_INFO(this->get_logger(), "Received Edge: id_ini=%d, id_end=%d, dx=%.2f, dy=%.2f, dtheta=%.2f",
    //         edge_id_ini, edge_id_end, edge_dx, edge_dy, edge_dth);
    // Send the received edge data to the processing thread
    std::lock_guard<std::mutex> lock(global_mutex_);
    edge_data_ = {edge_dx, edge_dy, edge_dth};
    edge_information_ = {msg->data[5], msg->data[6], msg->data[7], msg->data[8], msg->data[9], msg->data[10]};
    edge_data_ready_ = true;
    edge_data_condition_.notify_one();

    if (text_write) { // Check if text_write is enabled
      // Append edge_data to a txt file
      std::ofstream edge_file("my_edge_data.txt", std::ios::app);
      if (edge_file.is_open()) {
        edge_file << "EDGE_SE2" << " " << edge_id_ini << " " << edge_id_end << " " << edge_dx << " " << edge_dy << " " << edge_dth << " "
                  << std::get<0>(edge_information_) << " " << std::get<1>(edge_information_) << " " << std::get<2>(edge_information_) << " "
                  << std::get<3>(edge_information_) << " " << std::get<4>(edge_information_) << " " << std::get<5>(edge_information_) << "\n";
        edge_file.close();
        RCLCPP_INFO(this->get_logger(), "Edge data appended to edge_data.txt");
      } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to open edge_data.txt for appending");
      }
    }
  }

  void loop_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
  {
    loop_id_ini = msg->data[0]; // Store initial vertex id in the class variable
    loop_id_end = msg->data[1]; // Store final vertex id in the class variable
    loop_dx = msg->data[2]; // Store dx in the class variable
    loop_dy = msg->data[3]; // Store dy in the class variable
    loop_dth = msg->data[4]; // Store dtheta in the class variable
    // RCLCPP_INFO(this->get_logger(), "Received Loop: id_ini=%d, id_end=%d, dx=%.2f, dy=%.2f, dtheta=%.2f",
    //     loop_id_ini, loop_id_end, loop_dx, loop_dy, loop_dth);
    // Send the received loop data to the processing thread
    std::lock_guard<std::mutex> lock(global_mutex_);
    loop_data_ = {loop_dx, loop_dy, loop_dth};
    loop_information_ = {msg->data[5], msg->data[6], msg->data[7], msg->data[8], msg->data[9], msg->data[10]}; // Store information matrix in the class variable
    loop_data_ready_ = true;
    loop_data_condition_.notify_one();

    if (text_write) { // Check if text_write is enabled
      // Append loop_data to a txt file
      std::ofstream loop_file("my_loop_data.txt", std::ios::app);
      if (loop_file.is_open()) {
        loop_file << "EDGE_SE2" << " " << loop_id_ini << " " << loop_id_end << " " << loop_dx << " " << loop_dy << " " << loop_dth << " "
                  << std::get<0>(loop_information_) << " " << std::get<1>(loop_information_) << " " << std::get<2>(loop_information_) << " "
                  << std::get<3>(loop_information_) << " " << std::get<4>(loop_information_) << " " << std::get<5>(loop_information_) << "\n";
        loop_file.close();
        RCLCPP_INFO(this->get_logger(), "Loop data appended to loop_data.txt");
      } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to open loop_data.txt for appending");
      }
    }
  }

  void optimize_callback(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (msg->data) {
      RCLCPP_INFO(this->get_logger(), "\033[1;31mExternal optimization triggered.\033[0m");
      // Notify the optimization thread to start
      external_optimize = true;
    }
  }
};

int main(int argc, char** argv) {
  
  rclcpp::init(argc, argv);

  std::map<int, ceres::examples::Pose2d> poses;
  std::vector<ceres::examples::Constraint2d> constraints;

  ceres::Problem problem;

  auto node = std::make_shared<CeresSubscriber>(argc, argv, poses, constraints, problem);
  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
