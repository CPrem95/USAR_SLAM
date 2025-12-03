from optparse import OptionParser
import rclpy
from rclpy.node import Node
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from std_msgs.msg import Float32MultiArray
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as rot
import logging
from usar_slam.src import sar_funcs as sar
from scipy.io import savemat
import cv2
import gc
from sklearn.neighbors import NearestNeighbors
import beepy

'''Parallel processing'''
import multiprocessing as mp
from multiprocessing import Queue
from geometry_msgs.msg import Pose2D
import threading
from std_msgs.msg import Bool

plt.ion()
# Define ANSI escape codes for colors
class Colors:
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    DARK_GREEN = '\033[32m'
    ORANGE = '\033[38;5;208m'
    RESET = '\033[0m'

# **********************************************************************************************
class SAR(Node):
    def __init__(self, topic_name_L, topic_name_R):
        super().__init__('sar_gen')

        # PARAMETERS
        self.radar_r = 180
        self.d = 210
        self.mpp = 5e-3
        self.mph = 6e-3
        self.gamma = (math.pi - math.radians(30))/2
        self.D_1 = []
        self.D_2 = []
        self.samp_size = 920/143
        self.n_samp = 1500 # Number of samples
        self.max_qsize = 1000
        self.memory_size = 50000
        self.winsize = 200 # window size for nodes      # 250   | 200
        self.img_window = 300 # < 2*self.winsize      # 450   | 300
        self.loop_min_matches = 5

        self.fig_update_freq = int(self.winsize/10) # Update the figure every 25 steps
        self.half_fov = math.radians(30)

        # VARIABLES
        self.moving = False
        self.odom_init = False
        self.queue_id = 0
        self.cur_odom = [0, 0, 0]
        self.prev_odom = [0, 0, 0]
        self.terminate = False

        self.shared_optimized_poses = mp.Array('d', np.zeros(self.n_samp * 4)) # Shared memory for optimized poses
        self.optimized_event = mp.Event()

        self.cur_obs_radar_L = np.zeros(self.n_samp)
        self.cur_obs_radar_R = np.zeros(self.n_samp)

        self.mem_obs_radar_L = np.zeros([self.memory_size, self.n_samp])
        self.mem_obs_radar_R = np.zeros([self.memory_size, self.n_samp])
        self.mem_odom = np.zeros([self.memory_size, 4])

        self.data_left = {}
        self.data_right = {}

        self.dists = np.arange(0, self.n_samp) * self.samp_size

        # Initialize keypoints_descriptors as an empty list
        self.keypoints_descriptors = []
        self.odoms = np.empty((0, 3))  # Initialize as an empty numpy array with shape (0, 4)
        self.imgs_1_regions = [] # After positive SAR and blurring >>> Region 1
        self.imgs_2_regions = [] # After positive SAR and blurring >>> Region 2
        self.T_odom0_mem_1 = [] # Odom0 transformation matrices for region 1 w.r.t the image origin
        
        # Parameters for the SAR image
        self.res = 5 # Resolution in mm
        self.radar_range = [300, 2500] # Radar range in mm
        self.loop_nn_k = 15 # Loop closure find nearest neighbors k # 15
        self.loop_nn_r = 10000 # Loop closure find nearest neighbors radius
        self.loop_ignore_hist = 20 # Loop closure ignore history of recent # nodes ( >=5 to have enough neighbors) # 20
        self.ransac_thresh = 20.0 # RANSAC threshold

        self.sar_area_x = 33000 # SAR area in mm # con_labs_1: 25000 | con_labs_2: 26500 | next_lab_1:  15000 | next_lab_2: 33000
        self.sar_area_y = 35000 # SAR area in mm # con_labs_1: 15000 | con_labs_2: 22000 | next_lab_1:  12000 | next_lab_2: 35000

        self.sar_orig_x = 3000 # SAR origin in mm: 3000/res = 600  || origin at lower left corner (as usual)
        self.sar_orig_y = 20000 # SAR origin in mm: 2500/res = 500 # con_labs_1: 6000 | con_labs_2: 13000 | next_lab_1: 4000 | next_lab_2: 20000

        self.sar_end_y = self.sar_area_y - self.sar_orig_y
        # current_index = cart2index(-6.1, -5.1, ori_x, end_y, res)
        # print(current_index)
        self.pixels_x = int(self.sar_area_x/self.res)
        self.pixels_y = int(self.sar_area_y/self.res)

        print('resolution: ', self.res)
        print('all pixels_x: ', self.pixels_x)
        print('all pixels_y: ', self.pixels_y)

        # Define the range for x and y
        x = np.arange(-self.sar_orig_x + self.res/2, self.sar_area_x - self.sar_orig_x + self.res/2, self.res)
        y = np.arange(self.sar_area_y - self.sar_orig_y - self.res/2, - self.sar_orig_y - self.res/2, - self.res)

        # Create the meshgrid
        self.img_dist_X, self.img_dist_Y = np.meshgrid(x, y)

        self.radar_img_size = (self.pixels_y, self.pixels_x)
        # Initialize the raw images with zeros
        self.radar_img = np.zeros(self.radar_img_size, dtype=np.float32)
        self.empty_img = self.radar_img.copy()
        # self.radar_img_dist = np.zeros((2, self.pixels_y, self.pixels_x), dtype=np.float32) # Distance matrix: 2 for [x and y
        # self.radar_img_dist = self.find_dist_matrix(self.radar_img_dist, self.pixels_x, self.pixels_y, self.res)

        Fs = 23.328e9 # Sampling frequency
        fc = 7.29e9
        BW = 1.4e9
        frac_bw = BW/fc
        PRF = 14e6
        VTX = 0.6
        self.uwb_t, self.uwb_pulse = sar.generate_uwb_pulse(Fs, fc, frac_bw, PRF, VTX)

        # Parameters for Pose Graph Optimization
        self.odom_info = [50.0, 0.0, 0.0, 
                          50.0, 0.0,
                          50.0] # Upper triangular matrix of the information matrix
        self.loop_info = [44.721360, -0.000000, 0.000000, 
                          44.721360, 0.000000, 
                          44.721360] # Upper triangular matrix of the information matrix

        
        # pipes for the raw plotter
        self.parent_pipe_obs_radar_L, child_pipe_obs_radar_L = mp.Pipe() # parent (plot), child (plotter)
        self.parent_pipe_obs_radar_R, child_pipe_obs_radar_R = mp.Pipe()

        self.send_obs_radar_L = self.parent_pipe_obs_radar_L.send
        self.send_obs_radar_R = self.parent_pipe_obs_radar_R.send
        
        # Queue for the SAR
        self.obs_radar_L_queue = Queue(self.max_qsize)
        self.obs_radar_R_queue = Queue(self.max_qsize)
        self.odom_queue = Queue(self.max_qsize)

        self.empty_obs = np.zeros(self.n_samp, dtype=np.float32)

        # Pipe for the SAR image
        self.parent_pipe_sar_img_left, child_pipe_sar_img_left = mp.Pipe()
        self.parent_pipe_sar_img_right, child_pipe_sar_img_right = mp.Pipe()
        self.parent_pipe_terminate, child_pipe_terminate = mp.Pipe()

        self.send_sar_img_left = self.parent_pipe_sar_img_left.send
        self.send_sar_img_right = self.parent_pipe_sar_img_right.send
        self.send_terminate = self.parent_pipe_terminate.send
        
        self.subscription = self.create_subscription(
            Float32MultiArray,
            topic_name_L,
            self.obs_radar_L,
            1
        )
        self.subscription

        self.subscription = self.create_subscription(
            Float32MultiArray,
            topic_name_R,
            self.obs_radar_R,
            1
        )
        self.subscription

        self.subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            1
        )
        self.subscription

        self.subscription = self.create_subscription(
            Bool,
            'stop_slam',
            self.stop_slam,
            1
        )
        self.subscription

        self.subscription = self.create_subscription(
            Bool,
            'optimized_flag',
            self.optimized_flag_callback,
            1
        )

        self.subscription = self.create_subscription(
            Float32MultiArray,
            'optimized_poses',
            self.optimized_poses_callback,
            1
        )
        self.subscription

        # Publisher for poses || Odometries as Pose2D
        self.pose2d_publisher = self.create_publisher(Pose2D, 'pose', 10)

        # Publisher for the edges between the poses || translation between the poses as Float32MultiArray
        self.edges_publisher = self.create_publisher(Float32MultiArray, 'edge', 10)

        # Publisher for the loop closure edges between the poses || translation between the poses as Float32MultiArray
        self.loop_closure_publisher = self.create_publisher(Float32MultiArray, 'loop', 10)

        # Publisher for optimizing poses command
        self.optimize_poses_publisher = self.create_publisher(Bool, 'optimize', 10)

        # Communication between the threads and the processes
        self.parent_pipe_radar_img, child_pipe_radar_img = mp.Pipe()
        self.send_radar_img = self.parent_pipe_radar_img.send
        self.radar_img_queue = Queue(self.max_qsize)
        # **********************************************************************************************
        # Start the SAR processes >>> MULTI-PROCESSING
        self.sar_process = mp.Process(target=self.SAR_generation,
                                        args=(
                                            self.obs_radar_L_queue,
                                            self.obs_radar_R_queue,
                                            self.odom_queue),
                                       daemon=False)
        self.sar_process.start()

        # Start the SAR plotter process >>> MULTI-PROCESSING
        self.update_process = mp.Process(target=self.update_figure, 
                 args=(
                     child_pipe_radar_img,\
                     self.radar_img_queue
                 ), 
                 daemon=True)
        self.update_process.start()

        # Start the SAR image generation process for SLAM >>> MULTI-PROCESSING
        self.sar_slam_process = mp.Process(target=self.SAR_slam_img_generation,
                                                args=(
                                                    child_pipe_sar_img_left,
                                                    child_pipe_sar_img_right,
                                                    child_pipe_terminate))
        self.sar_slam_process.start()
                        
    """Callback function for the odometry"""
    def odom_callback(self, msg):
        cur_x = msg.pose.pose.position.x * 1000 # Convert to mm
        cur_y = msg.pose.pose.position.y * 1000 # Convert to mm``
        # Get robot velocity in mm/s
        vx = msg.twist.twist.linear.x
        vth = msg.twist.twist.angular.z

        cur_theta = rot.from_quat([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]).as_euler('zyx')[0]
        # cur_theta = cur_theta + math.pi/2
        if not self.odom_init:
            self.odom_init = True
            self.init_odom = [0.0, 0.0, 0.0, 0.0] # x, y, theta, displacement
            '''To consider the initial odometry as the reference point'''
            self.odom_queue.put(self.init_odom)
            self.obs_radar_L_queue.put(self.cur_obs_radar_L)
            self.obs_radar_R_queue.put(self.cur_obs_radar_R)
            ''''''
            self.get_logger().info(f"{Colors.YELLOW}Added to the queue_id: {self.queue_id}{Colors.RESET}")
            # self.get_logger().info(f"{Colors.YELLOW}Initial odometry: {self.init_odom}{Colors.RESET}")
        else:
            if vx != 0 and vth < 0.01: # Moving forward or backward
                self.cur_odom = [cur_x - self.init_odom[0], cur_y - self.init_odom[1], cur_theta - self.init_odom[2], 0]
                # self.get_logger().info(f"{Colors.YELLOW}Current odometry: {[cur_x, cur_y, cur_theta]}{Colors.RESET}")
                # self.get_logger().info(f"{Colors.YELLOW}Modified odometry: {self.cur_odom}{Colors.RESET}")

                self.displacement = np.hypot(self.cur_odom[0] - self.prev_odom[0], self.cur_odom[1] - self.prev_odom[1])
                self.moving = True

                print("Moving") 
                if self.displacement >= 3:
                    self.cur_odom[3] = self.displacement
                    self.prev_odom = self.cur_odom
                    # Use a single call to put multiple items into the queues
                    self.obs_radar_L_queue.put_nowait(self.cur_obs_radar_L)
                    self.obs_radar_R_queue.put_nowait(self.cur_obs_radar_R)
                    self.odom_queue.put_nowait(self.cur_odom)
                    self.queue_id += 1
                    self.get_logger().info(f"{Colors.YELLOW}Added to the queue_id: {self.queue_id}{Colors.RESET}")
                    
            elif vx < 0.01 and vth > 0.01: # Rotating
                self.cur_odom = [cur_x - self.init_odom[0], cur_y - self.init_odom[1], cur_theta - self.init_odom[2], 0]

                self.ang_displacement = abs(self.cur_odom[2] - self.prev_odom[2])
                self.moving = True
                print("Moving") 

                if self.ang_displacement >= 0.002: # 0.002 rad = 0.114 degrees
                    self.prev_odom = self.cur_odom
                    self.obs_radar_L_queue.put_nowait(self.empty_obs)
                    self.obs_radar_R_queue.put_nowait(self.empty_obs)
                    self.odom_queue.put_nowait(self.cur_odom)
                    self.queue_id += 1
                    self.get_logger().info(f"{Colors.ORANGE}Added to the queue_id (rot): {self.queue_id}{Colors.RESET}")

            else: # Not moving
                self.moving = False
                print(f"{Colors.RED}Not Moving{Colors.RESET}")

    """Subscription callback functions for the radar observations"""
    def obs_radar_L(self, msg):
        series = msg.data
        # series = np.abs(series)
        series[0] = 0
        if self.moving:
            # self.send_obs_radar_L(series)
            self.cur_obs_radar_L = series
    
    def obs_radar_2(self, msg):
        series = msg.data
        # series = np.abs(series)
        series[0] = 0
        if self.moving:
            # self.send_obs_radar2(series)
            self.cur_obs_radar_2 = series
    
    def obs_radar_3(self, msg):
        series = msg.data
        # series = np.abs(series)
        series[0] = 0
        if self.moving:
            # self.send_obs_radar3(series)
            self.cur_obs_radar_3 = series

    def obs_radar_R(self, msg):
        series = msg.data
        # series = np.abs(series)
        series[0] = 0
        if self.moving:
            # self.send_obs_radar_R(series)
            self.cur_obs_radar_R = series
    
    def stop_slam(self, msg):
        """Stop the SLAM process"""
        self.terminate = msg.data
        if self.terminate:
            # self.sar_process.terminate()
            # self.sar_slam_process.terminate()
            # self.update_process.terminate()
            self.odom_queue.put_nowait(-1) # Send a termination signal to the SAR generation process
            self.obs_radar_L_queue.put_nowait(self.empty_obs)
            self.obs_radar_R_queue.put_nowait(self.empty_obs)
            self.get_logger().info(f"{Colors.RED}Stopping SLAM process...{Colors.RESET}")

    def optimized_flag_callback(self, msg):
        self.optimized_flag = msg.data
        self.optimized_event.set()  # Signal that optimized poses are available
        # self.get_logger().info(f"Poses optimized: {self.optimized_flag}")

    def optimized_poses_callback(self, msg):
        # Reshape the 1D array into an n*4 array
        optimized_poses = np.array(msg.data).reshape(-1, 4)
        # Update the shared memory array
        self.shared_optimized_poses[:optimized_poses.size] = optimized_poses.flatten()
        # self.get_logger().info(f"Optimized poses received")

    """For the lambda function in the subscription""" 
    def obs_radar(self, msg, obs_var):
        if self.moving:
            obs_var = msg.data
            obs_var[0] = 0
        else:
            obs_var = None
    
    """Add the radar observations to the SAR image"""
    # Multiprocess #1
    # Sends the accumulated observations to generate the SAR image generation process via pipes
    def SAR_generation(self, obs_radar_L_queue, obs_radar_R_queue, odom_queue):
        step_id = 0
        self.sar_odom = [0, 0, 0]
        tmp_data_left = {"px": [], "py": [], "val": [], "odom": []}
        tmp_data_right = {"px": [], "py": [], "val": [], "odom": []}

        try:
            # Parallel processing using threads for left and right radar
            def process(obs_radar, add_func, tmp_data):
                rc_obs = sar.pulse_compression(obs_radar, self.uwb_pulse, False)
                add_func(odom, rc_obs, self.img_dist_X, self.img_dist_Y, self.radar_img, self.res, tmp_data, self.half_fov, self.sar_orig_x, self.sar_end_y, self.radar_r, self.samp_size)

            while True:
                if odom_queue.qsize() > 0:
                    odom = odom_queue.get()
                    obs_radar_L = obs_radar_L_queue.get()
                    obs_radar_R = obs_radar_R_queue.get()

                    self.mem_obs_radar_L[step_id, :] = obs_radar_L
                    self.mem_obs_radar_R[step_id, :] = obs_radar_R
                    self.mem_odom[step_id, :] = odom

                    if odom == -1:
                        self.get_logger().info(f"{Colors.RED}Terminating SAR generation process...{Colors.RESET}")
                        # self.parent_pipe_sar_img_left.send(tmp_data_left)
                        # self.parent_pipe_sar_img_right.send(tmp_data_right)
                        # time.sleep(0.1)
                        savemat('sar_memory.mat', {
                            'obs_radar_L': self.mem_obs_radar_L,
                            'obs_radar_R': self.mem_obs_radar_R,
                            'odom': self.mem_odom,
                        })
                        self.send_terminate(True)
                        time.sleep(0.1)
                        print("Sent terminate signal")
                        self.parent_pipe_sar_img_left.send(tmp_data_left)
                        self.parent_pipe_sar_img_right.send(tmp_data_right)
                        raise KeyboardInterrupt
                        # break

                    threads = [
                        threading.Thread(target=process, args=(obs_radar_L, sar.add_sar_radar_1, tmp_data_left)),
                        threading.Thread(target=process, args=(obs_radar_R, sar.add_sar_radar_2, tmp_data_right))
                    ]
                    for t in threads: t.start()
                    for t in threads: t.join()

                    # print("Radar observations added at odom: ", odom[:3])
                    self.get_logger().info(f"{Colors.CYAN}Processed queue_id: {step_id}{Colors.RESET}")

                    step_id += 1
                    self.get_logger().info(f"{Colors.BLUE}remaining qsize: {odom_queue.qsize()}{Colors.RESET}")

                    if step_id % 25 == 0:
                        # self.send_radar_img(self.radar_img)
                        self.radar_img_queue.put(self.radar_img)

                    if step_id % self.winsize == 0:
                        self.get_logger().info(f"{Colors.MAGENTA}Sending data for post-processing!{Colors.RESET}")
                        # self.radar_img = self.empty_img.copy()
                        self.parent_pipe_sar_img_left.send(tmp_data_left)
                        self.parent_pipe_sar_img_right.send(tmp_data_right)
                        # Pre-allocate new dicts instead of clearing old ones
                        tmp_data_left = {"px": [], "py": [], "val": [], "odom": []}
                        tmp_data_right = {"px": [], "py": [], "val": [], "odom": []}
                        # Run garbage collection
                        gc.collect()
                        # Use numpy's in-place fill for faster reset
                        self.radar_img.fill(0)

        except KeyboardInterrupt:
            self.get_logger().info(f"{Colors.RED}Keyboard Interrupt{Colors.RESET}")
            savemat('sar_data.mat', {'obs_radar_L': self.mem_obs_radar_L, 'obs_radar_R': self.mem_obs_radar_R, 'odom': self.mem_odom, 'img': self.radar_img})
            pos_img = self.radar_img + np.abs(self.radar_img)
            self.pos_sar_fig = plt.figure()
            self.pos_sar_ax = self.pos_sar_fig.add_subplot(111)
            self.pos_sar_ax.set_aspect('equal', adjustable='box')
            self.pos_sar_ax.set_xlabel(f'Pixels_x [1pixel = {self.res}mm]')
            self.pos_sar_ax.set_ylabel(f'Pixels_y [1pixel = {self.res}mm]')
            self.pos_sar_ax.set_title('Positive SAR output')
            self.pos_sar_img_obj = self.pos_sar_ax.imshow(pos_img, cmap='jet', animated=True)
            self.pos_sar_fig.canvas.draw()
            self.pos_sar_fig.canvas.flush_events()

            blurred = cv2.GaussianBlur(pos_img, (7, 7), 2) 
            self.blur_sar_fig = plt.figure()
            self.blur_sar_ax = self.blur_sar_fig.add_subplot(111)
            self.blur_sar_ax.set_aspect('equal', adjustable='box')
            self.blur_sar_ax.set_xlabel(f'Pixels_x [1pixel = {self.res}mm]')
            self.blur_sar_ax.set_ylabel(f'Pixels_y [1pixel = {self.res}mm]')
            self.blur_sar_ax.set_title('Blurred SAR output [online]')
            self.blur_sar_img_obj = self.blur_sar_ax.imshow(blurred, cmap='jet', animated=True)
            self.blur_sar_fig.canvas.draw()
            self.blur_sar_fig.canvas.flush_events()

            time.sleep(1000)
            pass
    
    """Generates sliced SAR images <local views> for SLAM"""
    # Multiprocess #2
    # Receives the radar images from the SAR generation process via pipes and processes them
    def SAR_slam_img_generation(self, pipe_left_img, pipe_right_img, pipe_terminate):
        data_left_flag = False 
        data_right_flag = False
        show_fig = False
        img_id = 0
        # print('SAR image size: ', self.radar_img_size)
        
        sar_imgs_fig = plt.figure()
        sar_ax1 = sar_imgs_fig.add_subplot(111)
        sar_ax1.set_aspect('equal', adjustable='box')
        sar_ax1.set_xlabel(f'Pixels_x [1pixel = {self.res}mm]')
        sar_ax1.set_ylabel(f'Pixels_y [1pixel = {self.res}mm]')
        sar_ax1.set_title('Intermed SAR out: R1')
        sar_imgs_fig.suptitle(f'Intermediate SAR outputs', color='black')
        radar_img_obj_1 = sar_ax1.imshow(np.zeros(self.radar_img_size), cmap='jet', animated=True)
        sar_imgs_fig.canvas.draw()
        sar_imgs_fig.canvas.flush_events()

        '''
        feature_fig = plt.figure()
        sift_ax1 = feature_fig.add_subplot(321)
        sift_ax2 = feature_fig.add_subplot(322)
        orb_ax1 = feature_fig.add_subplot(323)
        orb_ax2 = feature_fig.add_subplot(324)
        surf_ax1 = feature_fig.add_subplot(325)
        surf_ax2 = feature_fig.add_subplot(326)
        sift_ax1.set_aspect('equal', adjustable='box')
        sift_ax2.set_aspect('equal', adjustable='box')
        orb_ax1.set_aspect('equal', adjustable='box')
        orb_ax2.set_aspect('equal', adjustable='box')
        surf_ax1.set_aspect('equal', adjustable='box')
        surf_ax2.set_aspect('equal', adjustable='box')
        sift_ax1.set_title('SIFT: R1')
        sift_ax2.set_title('SIFT: R1&2')
        orb_ax1.set_title('ORB: R1')
        orb_ax2.set_title('ORB: R1&2')
        surf_ax1.set_title('SURF: R1')
        surf_ax2.set_title('SURF: R1&2')
        feature_fig.canvas.draw()
        feature_fig.canvas.flush_events()
        '''
        match_fig = plt.figure(figsize=(10, 5))
        # sift_ax = match_fig.add_subplot(131)
        # surf_ax = match_fig.add_subplot(132)
        akaze_ax = match_fig.add_subplot(121)
        orb_ax = match_fig.add_subplot(122)
        match_fig.suptitle(f'Keypoint Matching Across SAR Regions', fontsize=16, color='black')

        # sift_ax.set_aspect('equal', adjustable='box')
        # surf_ax.set_aspect('equal', adjustable='box')
        akaze_ax.set_aspect('equal', adjustable='box')
        orb_ax.set_aspect('equal', adjustable='box')
        # sift_ax.set_title('SIFT')
        # surf_ax.set_title('SURF')
        akaze_ax.set_title('AKAZE')
        orb_ax.set_title('ORB')
        match_fig.canvas.draw()
        match_fig.canvas.flush_events()

        tmp_imgs_1_regions = [] # Before positive SAR and blurring
        odoms = []
        loop_count = 0
        received_opt_poses = False

        while rclpy.ok():
            if pipe_left_img.poll():
                data_left = pipe_left_img.recv()
                px_left = data_left["px"]
                py_left = data_left["py"]
                val_left = data_left["val"]
                odom_left = data_left["odom"]
                data_left_flag = True

            if pipe_right_img.poll():    
                data_right = pipe_right_img.recv()
                px_right = data_right["px"]
                py_right = data_right["py"]
                val_right = data_right["val"]
                odom_right = data_right["odom"]
                data_right_flag = True

            if data_left_flag and data_right_flag:
                if self.optimized_event.is_set():
                    # Get the optimized poses from shared memory
                    optimized_poses = np.frombuffer(self.shared_optimized_poses.get_obj()).reshape((-1, 4))
                    mask = ~(optimized_poses == 0).all(axis=1)        # non-zero rows
                    last_idx = np.where(mask)[0].max()     # last non-zero row
                    optimized_poses = optimized_poses[:last_idx+1]
                    # self.get_logger().info(f"{Colors.BOLD}{Colors.GREEN}Optimized poses available: {optimized_poses}{Colors.RESET}")
                    self.optimized_event.clear()  # Reset the event
                    received_opt_poses = True

                self.get_logger().info(f"{Colors.MAGENTA}Processing SAR window ID: {img_id}{Colors.RESET}")
                gc.collect()
                data_left_flag = False 
                data_right_flag = False
                
                # Process the data here
                # Generate the SAR image
                for i in range(len(odom_left)):
                    self.radar_img[py_left[i], px_left[i]] += val_left[i]
                    self.radar_img[py_right[i], px_right[i]] += val_right[i]
                # Store the odoms
                # odoms.append([odom_left[0], odom_left[-1]])
                odoms += odom_left
                img_id += 1
                
                # add the TMP #1 REGION SAR image to the list of images
                tmp_imgs_1_regions.append(self.radar_img.copy())
                self.radar_img = self.empty_img.copy()

                """ Extract the SAR image regions """
                if img_id == 2:

                    img_0 = tmp_imgs_1_regions[0]
                    img_1 = tmp_imgs_1_regions[1]
                    
                    ''' add the #1 REGION SAR image to the list of images '''
                    sar_1 = img_0 + img_1
                    abs_sar_1 = np.abs(sar_1)
                    positive_sar_1 = sar_1 + abs_sar_1
                    blurred_sar_1 = cv2.GaussianBlur(positive_sar_1, (0, 0), 2)
                    odom0 = odoms[0]
                    odom1 = odoms[self.img_window]
                    min_x, max_x, min_y, max_y, T_odom0 = sar.extract_img_region(blurred_sar_1, self.res, odom0, odom1, self.radar_range[1]+500, self.sar_orig_x, self.sar_end_y) # extract_img_pixels(image, res, odom0, odom1, r_max, orig_x, end_y)
                    # self.get_logger().info(f"{Colors.RED}T_odom0_R1: {T_odom0}{Colors.RESET}")
                    print('limits: ', min_x, max_x, min_y, max_y)
                    cropped_sar_1 = blurred_sar_1[min_y:max_y, min_x:max_x]
                    cropped_sar_1 = cv2.normalize(cropped_sar_1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    self.imgs_1_regions.append(cropped_sar_1)   
                    self.T_odom0_mem_1.append(T_odom0)
                    sar_ax1.imshow(cropped_sar_1, cmap='jet', animated=True)   

                    sar_imgs_fig.suptitle(f'Intermediate SAR out ID: 0')
                    sar_imgs_fig.canvas.draw()
                    sar_imgs_fig.canvas.flush_events()
                    
                    ''' Publish pose to the graph '''
                    pose_msg = Pose2D()
                    pose_msg.x = odom0[0] / 1000  # Convert to meters
                    pose_msg.y = odom0[1] / 1000  # Convert to meters
                    pose_msg.theta = odom0[2]
                    self.pose2d_publisher.publish(pose_msg)
                    self.get_logger().info(f"{Colors.BOLD}{Colors.GREEN}Pose published: {pose_msg}{Colors.RESET}")

                    """ Feature extraction """
                    # # SIFT 
                    # sar.SIFT_extract_and_display_keypoints(cropped_sar_1, cropped_sar_2, sift_ax1, sift_ax2, feature_fig)
                    # # ORB
                    # sar.ORB_extract_and_display_keypoints(cropped_sar_1, cropped_sar_2, orb_ax1, orb_ax2, feature_fig)
                    # # SURF
                    # sar.SURF_extract_and_display_keypoints(cropped_sar_1, cropped_sar_2, surf_ax1, surf_ax2, feature_fig)

                    # SIFT
                    # kp_sift_1, des_sift_1, kp_sift_2, des_sift_2 = sar.SIFT_extract_keypoints2(cropped_sar_1, cropped_sar_2)
                    # ORB
                    kp_orb_1, des_orb_1 = sar.ORB_extract_keypoints(cropped_sar_1)
                    # SURF
                    # kp_surf_1, des_surf_1, kp_surf_2, des_surf_2 = sar.SURF_extract_keypoints2(cropped_sar_1, cropped_sar_2)
                    # AKAZE
                    kp_akaze_1, des_akaze_1 = sar.AKAZE_extract_keypoints(cropped_sar_1)

                    self.keypoints_descriptors.append({
                        # 'sift': {'kp1': kp_sift_1, 'kp2': kp_sift_2, 'des1': des_sift_1, 'des2': des_sift_2},
                        'orb': {'kp1': kp_orb_1, 'des1': des_orb_1},
                        # 'surf': {'kp1': kp_surf_1, 'kp2': kp_surf_2, 'des1': des_surf_1, 'des2': des_surf_2}
                        'akaze': {'kp1': kp_akaze_1, 'des1': des_akaze_1}
                    })
                    self.odoms = np.append(self.odoms, [odom0[:3]], axis=0)

                    # Save cropped SAR images in .mat format
                    savemat(f'cropped_sar_1_{0}.mat', {'cropped_sar_1': cropped_sar_1})
                    # Save the SAR images figure as an image file
                    sar_imgs_fig.savefig(f'sar_imgs_fig_reg_{0}.png')

                elif img_id > 2:
                    current_index = img_id - 2
                    ''' Publish edge to the graph '''
                    edge_msg = Float32MultiArray()
                    odom00 = odoms[0]
                    odom01 = odoms[self.winsize]
                    rot1 = np.arctan2(odom01[1] - odom00[1], odom01[0] - odom00[0]) - odom00[2]
                    tran1 = np.hypot(odom01[0] - odom00[0], odom01[1] - odom00[1])/ 1000 # Convert to meters
                    rot2 = odom01[2] - odom00[2] - rot1

                    dx = tran1 * np.cos(rot1)
                    dy = tran1 * np.sin(rot1)
                    dth = rot1 + rot2

                    edge_msg.data = [float(current_index -1), float(current_index), dx, dy, dth] + self.odom_info
                    self.edges_publisher.publish(edge_msg)  
                    self.get_logger().info(f"{Colors.BOLD}{Colors.GREEN}Edge published: {edge_msg}{Colors.RESET}") 

                    # Bought the below line here, otherwise the edge misses the first edge constraint
                    odoms = odoms[self.winsize:] # Remove the first odom set

                    ''' Publish pose to the graph '''
                    pose_msg = Pose2D()
                    odom0 = odoms[0]
                    pose_msg.x = odom0[0] / 1000  # Convert to meters
                    pose_msg.y = odom0[1] / 1000  # Convert to meters
                    pose_msg.theta = odom0[2]
                    self.pose2d_publisher.publish(pose_msg)
                    self.get_logger().info(f"{Colors.BOLD}{Colors.GREEN}Pose published: {pose_msg}{Colors.RESET}")

                    ''' add the #1 REGION SAR image to the list of images '''
                    img_0 = tmp_imgs_1_regions[0]
                    img_1 = tmp_imgs_1_regions[1]
                    img_2 = tmp_imgs_1_regions[2]

                    ''' add the #1 REGION SAR image to the list of images '''
                    sar_1 = img_0 + img_1 + img_2
                    abs_sar_1 = np.abs(sar_1)
                    positive_sar_1 = sar_1 + abs_sar_1
                    blurred_sar_1 = cv2.GaussianBlur(positive_sar_1, (0, 0), 2)
                    odom1 = odoms[self.img_window]
                    min_x, max_x, min_y, max_y, T_odom0 = sar.extract_img_region(blurred_sar_1, self.res, odom0, odom1, self.radar_range[1]+500, self.sar_orig_x, self.sar_end_y) # extract_img_pixels(image, res, odom0, odom1, r_max, orig_x, end_y)
                    # self.get_logger().info(f"{Colors.RED}T_odom0_R1: {T_odom0}{Colors.RESET}")
                    cropped_sar_1 = blurred_sar_1[min_y:max_y, min_x:max_x]
                    cropped_sar_1 = cv2.normalize(cropped_sar_1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    self.imgs_1_regions.append(cropped_sar_1)
                    self.T_odom0_mem_1.append(T_odom0)
                    sar_ax1.imshow(cropped_sar_1, cmap='jet', animated=True)
                    
                    sar_imgs_fig.suptitle(f'Intermed SAR out ID: {current_index}')
                    sar_imgs_fig.canvas.draw()
                    sar_imgs_fig.canvas.flush_events()   

                    """ Feature extraction """
                    # # SIFT
                    # sar.SIFT_extract_and_display_keypoints(cropped_sar_1, cropped_sar_2, sift_ax1, sift_ax2, feature_fig)
                    # # ORB
                    # sar.ORB_extract_and_display_keypoints(cropped_sar_1, cropped_sar_2, orb_ax1, orb_ax2, feature_fig)
                    # # SURF
                    # sar.SURF_extract_and_display_keypoints(cropped_sar_1, cropped_sar_2, surf_ax1, surf_ax2, feature_fig)

                    # SIFT
                    # kp_sift_1, des_sift_1, kp_sift_2, des_sift_2 = sar.SIFT_extract_keypoints2(cropped_sar_1, cropped_sar_2)
                    # ORB
                    kp_orb_1, des_orb_1 = sar.ORB_extract_keypoints(cropped_sar_1)
                    # SURF
                    # kp_surf_1, des_surf_1, kp_surf_2, des_surf_2 = sar.SURF_extract_keypoints2(cropped_sar_1, cropped_sar_2)
                    # AKAZE
                    kp_akaze_1, des_akaze_1 = sar.AKAZE_extract_keypoints(cropped_sar_1)

                    self.keypoints_descriptors.append({
                        # 'sift': {'kp1': kp_sift_1, 'kp2': kp_sift_2, 'des1': des_sift_1, 'des2': des_sift_2},
                        'orb': {'kp1': kp_orb_1, 'des1': des_orb_1},
                        # 'surf': {'kp1': kp_surf_1, 'kp2': kp_surf_2, 'des1': des_surf_1, 'des2': des_surf_2},
                        'akaze': {'kp1': kp_akaze_1, 'des1': des_akaze_1}
                    })

                    self.odoms = np.append(self.odoms, [odom0[:3]], axis=0)

                    """ Loop closure detection """
                    if current_index > self.loop_ignore_hist + self.loop_nn_k: # Only check for loop closure before the recent history 
                        # Ignore recent odoms and consider the rest for loop closure
                        if received_opt_poses:
                            n_poses = optimized_poses.shape[0]
                            # print("Number of optimized poses: ", n_poses)
                            non_optimized_odoms = self.odoms[n_poses - 1:]
                            # print("non-optimized odoms: ", non_optimized_odoms)

                            reference_odom = self.odoms[n_poses -1]
                            # print("reference_odom: ", reference_odom)
                            optimized_odom = optimized_poses[-1]
                            # print("optimized_odom: ", optimized_odom)

                            T0 = np.array([[np.cos(reference_odom[2]), -np.sin(reference_odom[2]), reference_odom[0]],
                                            [np.sin(reference_odom[2]), np.cos(reference_odom[2]), reference_odom[1]],
                                            [0, 0, 1]])
                            T0_inv = np.linalg.inv(T0)
                            
                            T1 = np.array([[np.cos(optimized_odom[3]), -np.sin(optimized_odom[3]), optimized_odom[1]*1000],
                                            [np.sin(optimized_odom[3]), np.cos(optimized_odom[3]), optimized_odom[2]*1000],
                                            [0, 0, 1]])

                            corrected_odoms = []
                            for i in range(len(non_optimized_odoms)):
                                odom = non_optimized_odoms[i]
                                # Apply the transformation
                                transformed_odom = T1 @ T0_inv @ np.array([odom[0], odom[1], 1]).T
                                # print("transformed_odom: ", transformed_odom)
                                corrected_odoms.append(transformed_odom[:2])

                            all_corrected_poses = np.append(optimized_poses[:, 1:3]*1000 , np.array(corrected_odoms), axis=0)
                            # print("all_corrected_poses: ", all_corrected_poses)

                            loop_poses = all_corrected_poses[:-self.loop_ignore_hist]
                            current_pose = all_corrected_poses[-1]

                        else:
                            loop_poses = self.odoms[:-self.loop_ignore_hist, :2]
                            current_pose = odom0[:2]
                        # print("loop_poses:", loop_poses)
                        # print("current_pose:", current_pose)

                        # Find nearest neighbors for loop closure
                        nn = NearestNeighbors(n_neighbors=self.loop_nn_k, radius=self.loop_nn_r, algorithm='auto')
                        nn.fit(loop_poses)
                        distances, indices = nn.kneighbors([current_pose]) # radius_neighbors does not consider the k nearest neighbors, it considers all neighbors within the radius

                        for i, index in enumerate(indices[0]):
                            distance = distances[0][i]
                            if distance < self.loop_nn_r: # If the distance is within the radius
                                self.get_logger().info(f"{Colors.RED}Feature Matching\nNear index: {index}, Current index: {current_index}, Distance: {distance}{Colors.RESET}")
                                # Display keypoints belonging to 'index'
                                tmp_cropped_sar_1 = self.imgs_1_regions[index] # loop closing img corresponding to the 'index' in region 1

                                # tmp_kp_sift_1 = self.keypoints_descriptors[index]['sift']['kp1']
                                # tmp_kp_sift_2 = self.keypoints_descriptors[index]['sift']['kp2']
                                tmp_kp_orb_1 = self.keypoints_descriptors[index]['orb']['kp1']
                                # tmp_kp_surf_1 = self.keypoints_descriptors[index]['surf']['kp1']
                                # tmp_kp_surf_2 = self.keypoints_descriptors[index]['surf']['kp2']
                                tmp_kp_akaze_1 = self.keypoints_descriptors[index]['akaze']['kp1']
                                # tmp_des_sift_1 = self.keypoints_descriptors[index]['sift']['des1']
                                # tmp_des_sift_2 = self.keypoints_descriptors[index]['sift']['des2']
                                tmp_des_orb_1 = self.keypoints_descriptors[index]['orb']['des1']
                                # tmp_des_surf_1 = self.keypoints_descriptors[index]['surf']['des1']
                                # tmp_des_surf_2 = self.keypoints_descriptors[index]['surf']['des2']
                                tmp_des_akaze_1 = self.keypoints_descriptors[index]['akaze']['des1']

                                match_fig.suptitle(f'Keypoint Matching Across SAR Regions [{index}, {current_index}]', fontsize=16)
                                
                                '''
                                des1 = tmp_des_sift_2 # loop closing img feature descriptors
                                des2 = des_sift_2 # current img feature descriptors
                                kp1 = tmp_kp_sift_2 # loop closing img keypoints
                                kp2 = kp_sift_2 # current img keypoints
                                if not (des1 is None or des2 is None):
                                    sift_matches = sar.match_kp(kp1, kp2, des1, des2, tmp_cropped_sar_2, cropped_sar_2, match_fig, sift_ax, False)
                                    sift_n_matches, sift_M, sift_scale, sift_rot, sift_disp = sar.find_homography(kp1, kp2, tmp_cropped_sar_2, cropped_sar_2, sift_matches, 'sift', self.ransac_thresh, match_fig, sift_ax, True)

                                des1 = tmp_des_surf_2 # loop closing img feature descriptors
                                des2 = des_surf_2 # current img feature descriptors
                                kp1 = tmp_kp_surf_2 # loop closing img keypoints
                                kp2 = kp_surf_2 # current img keypoints
                                if not (des1 is None or des2 is None):
                                    surf_matches = sar.match_kp(kp1, kp2, des1, des2, tmp_cropped_sar_2, cropped_sar_2, match_fig, surf_ax, False)
                                    surf_n_matches, surf_M, surf_scale, surf_rot, surf_disp = sar.find_homography(kp1, kp2, tmp_cropped_sar_2, cropped_sar_2, surf_matches, 'surf', self.ransac_thresh, match_fig, surf_ax, True)
                                '''
                                des1 = tmp_des_orb_1 # loop closing img feature descriptors
                                des2 = des_orb_1 # current img feature descriptors
                                kp1 = tmp_kp_orb_1 # loop closing img keypoints
                                kp2 = kp_orb_1 # current img keypoints
                                if not(des1 is None or des2 is None):
                                    orb_matches = sar.match_kp(kp1, kp2, des1, des2, tmp_cropped_sar_1, cropped_sar_1, match_fig, orb_ax, False)
                                    orb_matches_filt, orb_n_matches, orb_M, orb_scale, orb_rot, orb_disp = sar.find_homography(kp1, kp2, tmp_cropped_sar_1, cropped_sar_1, orb_matches, 'orb', self.ransac_thresh, match_fig, orb_ax, False)

                                des1 = tmp_des_akaze_1 # loop closing img feature descriptors
                                des2 = des_akaze_1 # current img feature descriptors
                                kp1 = tmp_kp_akaze_1 # loop closing img keypoints
                                kp2 = kp_akaze_1 # current img keypoints
                                if not(des1 is None or des2 is None):
                                    akaze_matches = sar.match_kp(kp1, kp2, des1, des2, tmp_cropped_sar_1, cropped_sar_1, match_fig, akaze_ax, False)
                                    akaze_matches_filt, akaze_n_matches, akaze_M, akaze_scale, akaze_rot, akaze_disp = sar.find_homography(kp1, kp2, tmp_cropped_sar_1, cropped_sar_1, akaze_matches, 'akaze', self.ransac_thresh, match_fig, akaze_ax, False)

                                # Check for loop closure
                                if all_in_region([akaze_scale, orb_scale], 0.8, 1.2) and within_same_region([akaze_rot, orb_rot], 0.1):
                                    if akaze_n_matches > self.loop_min_matches and orb_n_matches > self.loop_min_matches:
                                        loop_index = index
                                        self.get_logger().info(f"{Colors.BOLD}{Colors.RED}Loop Closure Detected!{Colors.RESET}")
                                        # beepy.beep(sound=1)

                                        """"
                                        des1 = tmp_des_orb_2 # loop closing img feature descriptors
                                        des2 = des_orb_2 # current img feature descriptors
                                        kp1 = tmp_kp_orb_2 # loop closing img keypoints
                                        kp2 = kp_orb_2 # current img keypoints
                                        orb_n_matches, orb_M, orb_scale, orb_rot, orb_disp = sar.find_homography(kp1, kp2, tmp_cropped_sar_2, cropped_sar_2, orb_matches, 'orb', self.ransac_thresh, match_fig, orb_ax, True)
                                        des1 = tmp_des_akaze_2 # loop closing img feature descriptors
                                        des2 = des_akaze_2 # current img feature descriptors
                                        kp1 = tmp_kp_akaze_2 # loop closing img keypoints
                                        kp2 = kp_akaze_2 # current img keypoints
                                        akaze_n_matches, akaze_M, akaze_scale, akaze_rot, akaze_disp = sar.find_homography(kp1, kp2, tmp_cropped_sar_2, cropped_sar_2, akaze_matches, 'akaze', self.ransac_thresh, match_fig, akaze_ax, True)
                                        """
                                        if True:
                                            img_orb = cv2.drawMatches(tmp_cropped_sar_1, tmp_kp_orb_1, cropped_sar_1, kp_orb_1, orb_matches_filt, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                                            orb_ax.clear()
                                            orb_ax.imshow(img_orb)
                                            img_akaze = cv2.drawMatches(tmp_cropped_sar_1, tmp_kp_akaze_1, cropped_sar_1, kp_akaze_1, akaze_matches_filt, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                                            akaze_ax.clear()
                                            akaze_ax.imshow(img_akaze)
                                            akaze_ax.set_title('AKAZE')
                                            orb_ax.set_title('ORB')
                                            match_fig.suptitle(f'Keypoint Matching Across SAR Regions [{loop_index}, {current_index}] \nLoop Closure Detected!!!', fontsize=20, color='red', fontweight='bold')
                                            match_fig.canvas.draw()
                                            match_fig.canvas.flush_events()
                                            match_fig.savefig(f'keypoint_matching_{loop_index}_{current_index}.png')
                                        
                                        # time.sleep(1)
                                        # Publish loop closure message e.g. 1183 1214 -0.990834 0.053848 0.002936 44.721360 -0.000000 0.000000 44.721360 0.000000 44.721360
                                        loop_msg = Float32MultiArray()
                                        
                                        _,  theta_, disp_ = get_weighted_M3(orb_disp, akaze_disp, orb_rot, akaze_rot, orb_n_matches, akaze_n_matches)
                                        dx_ = disp_[0]*self.res # x displacement [mm]
                                        dy_ = disp_[1]*self.res # y displacement [mm]
                                        T_match = np.array([[np.cos(theta_), -np.sin(theta_), dx_],
                                                            [np.sin(theta_), np.cos(theta_), dy_],
                                                            [0, 0, 1]])
                                        print(f"Loop closure T_match: {T_match}")

                                        # _, _, theta_, disp_, _ =  get_weighted_M2(sift_M, surf_M, orb_M, sift_n_matches, surf_n_matches, orb_n_matches)
                                        # dx_ = disp_[0]*self.res # x displacement [mm]
                                        # dy_ = disp_[1]*self.res # y displacement [mm]
                                        # T_match2 = np.array([[np.cos(theta_), -np.sin(theta_), dx_],
                                        #                     [np.sin(theta_), np.cos(theta_), dy_],
                                        #                     [0, 0, 1]])
                                        # print(f"Loop closure T_match2: {T_match2}")

                                        T_odom0_1 = self.T_odom0_mem_1[loop_index] # [mm]
                                        T_odom0_2 = self.T_odom0_mem_1[current_index] # [mm]

                                        T_1_R2 = np.linalg.inv(T_match) @ T_odom0_2 # Transformation from R2 to frame_1
                                        T_R1_R2 = np.linalg.inv(T_odom0_1) @ T_1_R2 # Transformation from R1 to R2
                                        print(f"Loop closure T_odom0_1: {T_odom0_1}\nT_odom0_2: {T_odom0_2}")
                                        print(f"Loop closure T_1_R2: {T_1_R2}\nT_R1_R2: {T_R1_R2}")

                                        dx_loop = T_R1_R2[0, 2]/1000 # x displacement [mm]
                                        dy_loop = T_R1_R2[1, 2]/1000 # y displacement [mm]
                                        theta_loop = np.arctan2(T_R1_R2[1, 0], T_R1_R2[0, 0]) # [rad]

                                        loop_msg.data = [float(loop_index), float(current_index), dx_loop, -dy_loop, -theta_loop] + self.loop_info
                                        self.loop_closure_publisher.publish(loop_msg)
                                        self.get_logger().info(f"{Colors.BOLD}{Colors.GREEN}Loop Closure Message published: {loop_msg}{Colors.RESET}")

                                        loop_count += 1

                                        if loop_count % 5 == 0:
                                            # Publish the optimize poses command
                                            self.optimize_poses_publisher.publish(Bool(data=True))
                                            self.get_logger().info(f"{Colors.BOLD}{Colors.GREEN}Optimize Poses Command published{Colors.RESET}")

                                        savemat(f'loop_closure_{loop_index}_{current_index}.mat', {
                                            'T_odom0_1': T_odom0_1,
                                            'T_odom0_2': T_odom0_2,
                                            'T_match': T_match,
                                            'T_1_R2': T_1_R2,
                                            'T_R1_R2': T_R1_R2,
                                            'orb_M': orb_M,
                                            'akaze_M': akaze_M,
                                            'orb_n_matches': orb_n_matches,
                                            'akaze_n_matches': akaze_n_matches,
                                        })
                                    # time.sleep(0.5)

                    tmp_imgs_1_regions.pop(0) # Remove the first image

                    # Save cropped SAR images in .mat format
                    savemat(f'cropped_sar_1_{current_index}.mat', {'cropped_sar_1': cropped_sar_1})
                    # Save the SAR images figure as an image file   
                    sar_imgs_fig.savefig(f'sar_imgs_fig_reg_{current_index}.png')

                    if show_fig:
                        # radar_img_obj.set_data(blurred_sar_1) 
                        # radar_img_obj.set_clim(vmin=np.min(blurred_sar_1), vmax=np.max(blurred_sar_1))
                        sar_imgs_fig.canvas.draw()
                        sar_imgs_fig.canvas.flush_events()

                # self.get_logger().info(f"{Colors.MAGENTA}Processed SAR window ID: {img_id -1}{Colors.RESET}")

            """    Check if the termination signal is received    """
            """ If the termination signal is received, generate the final SAR images and publish the pose and edges """
            if pipe_terminate.poll():
                terminate = pipe_terminate.recv()
                self.get_logger().info(f"{Colors.RED}Generating final SAR image(s)...{Colors.RESET}")
                while terminate:
                    if pipe_left_img.poll():
                        data_left = pipe_left_img.recv()
                        px_left = data_left["px"]
                        py_left = data_left["py"]
                        val_left = data_left["val"]
                        odom_left = data_left["odom"]
                        data_left_flag = True

                    if pipe_right_img.poll():    
                        data_right = pipe_right_img.recv()
                        px_right = data_right["px"]
                        py_right = data_right["py"]
                        val_right = data_right["val"]
                        odom_right = data_right["odom"]
                        data_right_flag = True

                    if data_left_flag and data_right_flag:
                        terminate = False
                        data_left_flag = False 
                        data_right_flag = False
                        
                        # Process the data here
                        # Generate the SAR image
                        for i in range(len(odom_left)):
                            self.radar_img[py_left[i], px_left[i]] += val_left[i]
                            self.radar_img[py_right[i], px_right[i]] += val_right[i]
                        # Store the odoms
                        # odoms.append([odom_left[0], odom_left[-1]])
                        odoms += odom_left
                        
                        # add the TMP #1 REGION SAR image to the list of images
                        tmp_imgs_1_regions.append(self.radar_img.copy())
                        self.radar_img = self.empty_img.copy()

                        #
                        # There is only one more pending image to process
                        #
                        self.get_logger().info(f"{Colors.MAGENTA}Processing SAR window ID: {-2}{Colors.RESET}")
                        current_index = img_id - 1
                        ''' Publish edge to the graph '''
                        edge_msg = Float32MultiArray()
                        odom00 = odoms[0]
                        odom01 = odoms[self.winsize]
                        rot1 = np.arctan2(odom01[1] - odom00[1], odom01[0] - odom00[0]) - odom00[2]
                        tran1 = np.hypot(odom01[0] - odom00[0], odom01[1] - odom00[1])/ 1000 # Convert to meters
                        rot2 = odom01[2] - odom00[2] - rot1

                        dx = tran1 * np.cos(rot1)
                        dy = tran1 * np.sin(rot1)
                        dth = rot1 + rot2

                        edge_msg.data = [float(current_index -1), float(current_index), dx, dy, dth] + self.odom_info
                        self.edges_publisher.publish(edge_msg)  
                        self.get_logger().info(f"{Colors.BOLD}{Colors.GREEN}Edge published: {edge_msg}{Colors.RESET}") 

                        # Bought the below line here, otherwise the edge misses the first edge constraint
                        odoms = odoms[self.winsize:] # Remove the first odom set

                        ''' Publish pose to the graph '''
                        pose_msg = Pose2D()
                        odom0 = odoms[0]
                        pose_msg.x = odom0[0] / 1000  # Convert to meters
                        pose_msg.y = odom0[1] / 1000  # Convert to meters
                        pose_msg.theta = odom0[2]
                        self.pose2d_publisher.publish(pose_msg)
                        self.get_logger().info(f"{Colors.BOLD}{Colors.GREEN}Pose published: {pose_msg}{Colors.RESET}")

                        ''' add the #1 REGION SAR image to the list of images '''
                        img_0 = tmp_imgs_1_regions[0]
                        img_1 = tmp_imgs_1_regions[1]
                        img_2 = tmp_imgs_1_regions[2]
                        print(f"len(tmp_imgs_1_regions): {len(tmp_imgs_1_regions)}")

                        ''' add the #1 REGION SAR image to the list of images '''
                        sar_1 = img_0 + img_1 + img_2
                        abs_sar_1 = np.abs(sar_1)
                        positive_sar_1 = sar_1 + abs_sar_1
                        blurred_sar_1 = cv2.GaussianBlur(positive_sar_1, (0, 0), 2)

                        print("len(odoms): ", len(odoms))
                        odoms_length = len(odoms)
                        if self.img_window <= odoms_length - 1:
                            odom1 = odoms[self.img_window]
                        else:
                            odom1 = odoms[-1]
                        min_x, max_x, min_y, max_y, T_odom0 = sar.extract_img_region(blurred_sar_1, self.res, odom0, odom1, self.radar_range[1]+500, self.sar_orig_x, self.sar_end_y) # extract_img_pixels(image, res, odom0, odom1, r_max, orig_x, end_y)
                        # self.get_logger().info(f"{Colors.RED}T_odom0_R1: {T_odom0}{Colors.RESET}")
                        cropped_sar_1 = blurred_sar_1[min_y:max_y, min_x:max_x]
                        cropped_sar_1 = cv2.normalize(cropped_sar_1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        self.imgs_1_regions.append(cropped_sar_1)
                        self.T_odom0_mem_1.append(T_odom0)
                        sar_ax1.imshow(cropped_sar_1, cmap='jet', animated=True)

                        sar_imgs_fig.suptitle(f'Intermed SAR out ID: {current_index}')
                        sar_imgs_fig.canvas.draw()
                        sar_imgs_fig.canvas.flush_events()   

                        """ Feature extraction """
                        # ORB
                        kp_orb_1, des_orb_1 = sar.ORB_extract_keypoints(cropped_sar_1)
                        # AKAZE
                        kp_akaze_1, des_akaze_1 = sar.AKAZE_extract_keypoints(cropped_sar_1)

                        self.keypoints_descriptors.append({
                            'orb': {'kp1': kp_orb_1, 'des1': des_orb_1},
                            'akaze': {'kp1': kp_akaze_1, 'des1': des_akaze_1}
                        })

                        self.odoms = np.append(self.odoms, [odom0[:3]], axis=0)

                        """ Loop closure detection """
                        if current_index > self.loop_ignore_hist + self.loop_nn_k: # Only check for loop closure before the recent history
                            # Ignore recent odoms and consider the rest for loop closure
                            if received_opt_poses:
                                n_poses = optimized_poses.shape[0]
                                # print("Number of optimized poses: ", n_poses)
                                non_optimized_odoms = self.odoms[n_poses - 1:]
                                # print("non-optimized odoms: ", non_optimized_odoms)

                                reference_odom = self.odoms[n_poses -1]
                                # print("reference_odom: ", reference_odom)
                                optimized_odom = optimized_poses[-1]
                                # print("optimized_odom: ", optimized_odom)

                                T0 = np.array([[np.cos(reference_odom[2]), -np.sin(reference_odom[2]), reference_odom[0]],
                                                [np.sin(reference_odom[2]), np.cos(reference_odom[2]), reference_odom[1]],
                                                [0, 0, 1]])
                                T0_inv = np.linalg.inv(T0)
                                
                                T1 = np.array([[np.cos(optimized_odom[3]), -np.sin(optimized_odom[3]), optimized_odom[1]*1000],
                                                [np.sin(optimized_odom[3]), np.cos(optimized_odom[3]), optimized_odom[2]*1000],
                                                [0, 0, 1]])

                                corrected_odoms = []
                                for i in range(len(non_optimized_odoms)):
                                    odom = non_optimized_odoms[i]
                                    # Apply the transformation
                                    transformed_odom = T1 @ T0_inv @ np.array([odom[0], odom[1], 1]).T
                                    # print("transformed_odom: ", transformed_odom)
                                    corrected_odoms.append(transformed_odom[:2])

                                all_corrected_poses = np.append(optimized_poses[:, 1:3]*1000 , np.array(corrected_odoms), axis=0)
                                # print("all_corrected_poses: ", all_corrected_poses)

                                loop_poses = all_corrected_poses[:-self.loop_ignore_hist]
                                current_pose = all_corrected_poses[-1]
                            else:
                                loop_poses = self.odoms[:-self.loop_ignore_hist, :2]
                                current_pose = odom0[:2]
                            # print("loop_poses:", loop_poses)
                            # print("current_pose:", current_pose)

                            # Find nearest neighbors for loop closure
                            nn = NearestNeighbors(n_neighbors=self.loop_nn_k, radius=self.loop_nn_r, algorithm='auto')
                            nn.fit(loop_poses)
                            distances, indices = nn.kneighbors([current_pose]) # radius_neighbors does not consider the k nearest neighbors, it considers all neighbors within the radius

                            for i, index in enumerate(indices[0]):
                                distance = distances[0][i]
                                if distance < self.loop_nn_r: # If the distance is within the radius
                                    self.get_logger().info(f"{Colors.RED}Feature Matching\nNear index: {index}, Current index: {current_index}, Distance: {distance}{Colors.RESET}")
                                    # Display keypoints belonging to 'index'
                                    tmp_cropped_sar_1 = self.imgs_1_regions[index] # loop closing img corresponding to the 'index' in region 1

                                    tmp_kp_orb_1 = self.keypoints_descriptors[index]['orb']['kp1']
                                    tmp_kp_akaze_1 = self.keypoints_descriptors[index]['akaze']['kp1']
                                    tmp_des_orb_1 = self.keypoints_descriptors[index]['orb']['des1']
                                    tmp_des_akaze_1 = self.keypoints_descriptors[index]['akaze']['des1']

                                    match_fig.suptitle(f'Keypoint Matching Across SAR Regions [{index}, {current_index}]', fontsize=16)

                                    des1 = tmp_des_orb_1 # loop closing img feature descriptors
                                    des2 = des_orb_1 # current img feature descriptors
                                    kp1 = tmp_kp_orb_1 # loop closing img keypoints
                                    kp2 = kp_orb_1 # current img keypoints
                                    if not(des1 is None or des2 is None):
                                        orb_matches = sar.match_kp(kp1, kp2, des1, des2, tmp_cropped_sar_1, cropped_sar_1, match_fig, orb_ax, False)
                                        orb_matches_filt, orb_n_matches, orb_M, orb_scale, orb_rot, orb_disp = sar.find_homography(kp1, kp2, tmp_cropped_sar_1, cropped_sar_1, orb_matches, 'orb', self.ransac_thresh, match_fig, orb_ax, False)

                                    des1 = tmp_des_akaze_1 # loop closing img feature descriptors
                                    des2 = des_akaze_1 # current img feature descriptors
                                    kp1 = tmp_kp_akaze_1 # loop closing img keypoints
                                    kp2 = kp_akaze_1 # current img keypoints
                                    if not(des1 is None or des2 is None):
                                        akaze_matches = sar.match_kp(kp1, kp2, des1, des2, tmp_cropped_sar_1, cropped_sar_1, match_fig, akaze_ax, False)
                                        akaze_matches_filt, akaze_n_matches, akaze_M, akaze_scale, akaze_rot, akaze_disp = sar.find_homography(kp1, kp2, tmp_cropped_sar_1, cropped_sar_1, akaze_matches, 'akaze', self.ransac_thresh, match_fig, akaze_ax, False)

                                    # Check for loop closure
                                    if all_in_region([akaze_scale, orb_scale], 0.8, 1.2) and within_same_region([akaze_rot, orb_rot], 0.1):
                                        if akaze_n_matches > self.loop_min_matches and orb_n_matches > self.loop_min_matches:
                                            loop_index = index
                                            self.get_logger().info(f"{Colors.BOLD}{Colors.RED}Loop Closure Detected!{Colors.RESET}")
                                            # beep(sound=1)
                                            """
                                            des1 = tmp_des_orb_2 # loop closing img feature descriptors
                                            des2 = des_orb_2 # current img feature descriptors
                                            kp1 = tmp_kp_orb_2 # loop closing img keypoints
                                            kp2 = kp_orb_2 # current img keypoints
                                            orb_n_matches, orb_M, orb_scale, orb_rot, orb_disp = sar.find_homography(kp1, kp2, tmp_cropped_sar_2, cropped_sar_2, orb_matches, 'orb', self.ransac_thresh, match_fig, orb_ax, True)
                                            des1 = tmp_des_akaze_2 # loop closing img feature descriptors
                                            des2 = des_akaze_2 # current img feature descriptors
                                            kp1 = tmp_kp_akaze_2 # loop closing img keypoints
                                            kp2 = kp_akaze_2 # current img keypoints
                                            akaze_n_matches, akaze_M, akaze_scale, akaze_rot, akaze_disp = sar.find_homography(kp1, kp2, tmp_cropped_sar_2, cropped_sar_2, akaze_matches, 'akaze', self.ransac_thresh, match_fig, akaze_ax, True)
                                            """
                                            if True:
                                                img_orb = cv2.drawMatches(tmp_cropped_sar_1, tmp_kp_orb_1, cropped_sar_1, kp_orb_1, orb_matches_filt, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                                                orb_ax.clear()
                                                orb_ax.imshow(img_orb)
                                                img_akaze = cv2.drawMatches(tmp_cropped_sar_1, tmp_kp_akaze_1, cropped_sar_1, kp_akaze_1, akaze_matches_filt, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                                                akaze_ax.clear()
                                                akaze_ax.imshow(img_akaze)
                                                akaze_ax.set_title('AKAZE')
                                                orb_ax.set_title('ORB')
                                                match_fig.suptitle(f'Keypoint Matching Across SAR Regions [{loop_index}, {current_index}] \nLoop Closure Detected!!!', fontsize=20, color='red', fontweight='bold')
                                                match_fig.canvas.draw()
                                                match_fig.canvas.flush_events()
                                                match_fig.savefig(f'keypoint_matching_{loop_index}_{current_index}.png')

                                            # time.sleep(1)
                                            # Publish loop closure message e.g. 1183 1214 -0.990834 0.053848 0.002936 44.721360 -0.000000 0.000000 44.721360 0.000000 44.721360
                                            loop_msg = Float32MultiArray()
                                            
                                            _,  theta_, disp_ = get_weighted_M3(orb_disp, akaze_disp, orb_rot, akaze_rot, orb_n_matches, akaze_n_matches)
                                            dx_ = disp_[0]*self.res # x displacement [mm]
                                            dy_ = disp_[1]*self.res # y displacement [mm]
                                            T_match = np.array([[np.cos(theta_), -np.sin(theta_), dx_],
                                                                [np.sin(theta_), np.cos(theta_), dy_],
                                                                [0, 0, 1]])
                                            print(f"Loop closure T_match: {T_match}")

                                            T_odom0_1 = self.T_odom0_mem_1[loop_index] # [mm]
                                            T_odom0_2 = self.T_odom0_mem_1[current_index] # [mm]

                                            T_1_R2 = np.linalg.inv(T_match) @ T_odom0_2 # Transformation from R2 to frame_1
                                            T_R1_R2 = np.linalg.inv(T_odom0_1) @ T_1_R2 # Transformation from R1 to R2
                                            print(f"Loop closure T_odom0_1: {T_odom0_1}\nT_odom0_2: {T_odom0_2}")
                                            print(f"Loop closure T_1_R2: {T_1_R2}\nT_R1_R2: {T_R1_R2}")

                                            dx_loop = T_R1_R2[0, 2]/1000 # x displacement [mm]
                                            dy_loop = T_R1_R2[1, 2]/1000 # y displacement [mm]
                                            theta_loop = np.arctan2(T_R1_R2[1, 0], T_R1_R2[0, 0]) # [rad]

                                            loop_msg.data = [float(loop_index), float(current_index), dx_loop, -dy_loop, -theta_loop] + self.loop_info
                                            self.get_logger().info(f"{Colors.BOLD}{Colors.GREEN}Loop Closure Message published: {loop_msg}{Colors.RESET}")

                                            savemat(f'loop_closure_{loop_index}_{current_index}.mat', {
                                                'T_odom0_1': T_odom0_1,
                                                'T_odom0_2': T_odom0_2,
                                                'T_match': T_match,
                                                'T_1_R2': T_1_R2,
                                                'T_R1_R2': T_R1_R2,
                                                'orb_M': orb_M,
                                                'akaze_M': akaze_M,
                                                'orb_n_matches': orb_n_matches,
                                                'akaze_n_matches': akaze_n_matches,
                                            })
                                            self.loop_closure_publisher.publish(loop_msg)
                                        # time.sleep(0.5)

                        # Save cropped SAR images in .mat format
                        savemat(f'cropped_sar_1_{current_index}.mat', {'cropped_sar_1': cropped_sar_1})
                        # Save the SAR images figure as an image file   
                        sar_imgs_fig.savefig(f'sar_imgs_fig_reg_{current_index}.png')

                        """"""
                        """ Publish final two poses """
                        """"""
                        self.get_logger().info(f"{Colors.RED}Publishing final node(s)...{Colors.RESET}")

                        ## Pose [-1]
                        self.get_logger().info(f"{Colors.MAGENTA}Processing SAR window ID: {-1}{Colors.RESET}")
                        current_index = img_id
                        ''' Publish edge to the graph '''
                        edge_msg = Float32MultiArray()
                        odom00 = odoms[0]
                        odom01 = odoms[-1] # Use the last odom for the last image
                        rot1 = np.arctan2(odom01[1] - odom00[1], odom01[0] - odom00[0]) - odom00[2]
                        tran1 = np.hypot(odom01[0] - odom00[0], odom01[1] - odom00[1])/ 1000 # Convert to meters
                        rot2 = odom01[2] - odom00[2] - rot1

                        dx = tran1 * np.cos(rot1)
                        dy = tran1 * np.sin(rot1)
                        dth = rot1 + rot2

                        edge_msg.data = [float(current_index -1), float(current_index), dx, dy, dth] + self.odom_info
                        self.edges_publisher.publish(edge_msg)  
                        self.get_logger().info(f"{Colors.BOLD}{Colors.GREEN}Edge published: {edge_msg}{Colors.RESET}") 
                        ''' Publish pose to the graph '''
                        pose_msg = Pose2D()
                        odom0 = odoms[-1]
                        pose_msg.x = odom0[0] / 1000  # Convert to meters
                        pose_msg.y = odom0[1] / 1000  # Convert to meters
                        pose_msg.theta = odom0[2]
                        self.pose2d_publisher.publish(pose_msg)
                        self.get_logger().info(f"{Colors.BOLD}{Colors.GREEN}Pose published: {pose_msg}{Colors.RESET}")
                        self.odoms = np.append(self.odoms, [odom0[:3]], axis=0)

                        print(f"{Colors.CYAN}{Colors.BOLD}Final odoms: {self.odoms}{Colors.RESET}")
                        print(f"{Colors.BOLD}{Colors.RED}Now optimize the poses!{Colors.RESET}")

                        if show_fig:
                            # radar_img_obj.set_data(blurred_sar_1) 
                            # radar_img_obj.set_clim(vmin=np.min(blurred_sar_1), vmax=np.max(blurred_sar_1))
                            sar_imgs_fig.canvas.draw()
                            sar_imgs_fig.canvas.flush_events()

                        time.sleep(20) # Wait for the figure to update
                        break
    '''
    SAR image update function
    This function is called in a separate process to update the SAR image
    '''
    # Multiprocess #3
    def update_figure(self, pipe, radar_img_queue):
        # If image hasn't been initialized, use imshow
        self.sar_fig_1 = plt.figure()
        self.sar_ax1 = self.sar_fig_1.add_subplot(121)
        self.sar_ax2 = self.sar_fig_1.add_subplot(122)
        self.sar_fig_2 = plt.figure()
        self.sar_ax3 = self.sar_fig_2.add_subplot(111)

        self.sar_ax1.set_aspect('equal', adjustable='box')
        self.sar_ax1.set_xlabel(f'Pixels_x [1pixel = {self.res}mm]')
        self.sar_ax1.set_ylabel(f'Pixels_y [1pixel = {self.res}mm]')
        self.sar_ax1.set_title('SAR output')

        self.sar_ax2.set_aspect('equal', adjustable='box')
        self.sar_ax2.set_xlabel(f'Pixels_x [1pixel = {self.res}mm]')
        self.sar_ax2.set_ylabel(f'Pixels_y [1pixel = {self.res}mm]')
        self.sar_ax2.set_title('Positive image')

        self.sar_ax3.set_aspect('equal', adjustable='box')
        self.sar_ax3.set_xlabel(f'Pixels_x [1pixel = {self.res}mm]')
        self.sar_ax3.set_ylabel(f'Pixels_y [1pixel = {self.res}mm]')
        self.sar_ax3.set_title('Filtered SAR image [online update]')
        self.radar_img_obj_1 = self.sar_ax1.imshow(np.zeros(self.radar_img_size), cmap='jet', animated=True)
        self.radar_img_obj_2 = self.sar_ax2.imshow(np.zeros(self.radar_img_size), cmap='jet', animated=True)
        self.radar_img_obj_3 = self.sar_ax3.imshow(np.zeros(self.radar_img_size), cmap='jet', animated=True)

        self.sar_fig_1.canvas.draw()
        self.sar_fig_1.canvas.flush_events()
        self.sar_fig_2.canvas.draw()
        self.sar_fig_2.canvas.flush_events()
        while True:
            # if pipe.poll():
            #     radar_img = pipe.recv()
            if not self.radar_img_queue.empty():
                radar_img = self.radar_img_queue.get()

                abs_sar = np.abs(radar_img)
                self.radar_img_obj_1.set_data(abs_sar)
                self.radar_img_obj_1.set_clim(vmin=np.min(abs_sar), vmax=np.max(abs_sar))
                positive_sar = radar_img + abs_sar
                self.radar_img_obj_2.set_data(abs_sar)
                self.radar_img_obj_2.set_clim(vmin=np.min(abs_sar), vmax=np.max(abs_sar))
                positive_sar_blur= cv2.GaussianBlur(positive_sar, (0, 0), 2)
                # positive_sar_blur = (positive_sar_blur - np.min(positive_sar_blur)) / (np.max(positive_sar_blur) - np.min(positive_sar_blur)) # Normalize the image
                self.radar_img_obj_3.set_data(positive_sar_blur)
                self.radar_img_obj_3.set_clim(vmin=np.min(positive_sar_blur), vmax=np.max(positive_sar_blur))

                self.sar_fig_1.canvas.draw()
                self.sar_fig_1.canvas.flush_events()
                self.sar_fig_2.canvas.draw()
                self.sar_fig_2.canvas.flush_events()

    def find_dist_matrix(self, img_dist, pixels_x, pixels_y, res):
        for i in range(0, pixels_y):
            pix_y = res*(pixels_y - i) - res/2
            for j in range(0, pixels_x):
                pix_x = j*res + res/2
                img_dist[0, i, j] = pix_x
                img_dist[1, i, j] = pix_y
        return img_dist

def within_same_region(numbers, tolerance):
    return max(numbers) - min(numbers) <= 2 * tolerance

def all_in_region(numbers, a, b, inclusive=True):
    if inclusive:
        return all(a <= x <= b for x in numbers)
    else:
        return all(a < x < b for x in numbers)

def get_weighted_M(sift_disp, surf_disp, orb_disp, sift_rot, surf_rot, orb_rot, sift_n_matches, surf_n_matches, orb_n_matches):
    # Calculate weights based on the number of matches
    total_matches = sift_n_matches + surf_n_matches + orb_n_matches
    if total_matches == 0:
        return np.eye(3)  # Return identity matrix if no matches

    sift_weight = sift_n_matches / total_matches
    surf_weight = surf_n_matches / total_matches
    orb_weight = orb_n_matches / total_matches

    weighted_disp = (sift_weight * sift_disp) + (surf_weight * surf_disp) + (orb_weight * orb_disp)
    weighted_rot = (sift_weight * sift_rot) + (surf_weight * surf_rot) + (orb_weight * orb_rot)
    weighted_M = np.array([[np.cos(weighted_rot), -np.sin(weighted_rot), weighted_disp[0]],
                            [np.sin(weighted_rot), np.cos(weighted_rot), weighted_disp[1]],
                            [0, 0, 1]])
    
    return weighted_M, weighted_rot, weighted_disp

def get_weighted_M2(sift_M, surf_M, orb_M, sift_n_matches, surf_n_matches, orb_n_matches):
    # Calculate weights based on the number of matches
    total_matches = sift_n_matches + surf_n_matches + orb_n_matches
    if total_matches == 0:
        return np.eye(3)  # Return identity matrix if no matches

    sift_weight = sift_n_matches / total_matches
    surf_weight = surf_n_matches / total_matches
    orb_weight = orb_n_matches / total_matches

    # Calculate the weighted average of the matrices
    weighted_M = (sift_weight * sift_M) + (surf_weight * surf_M) + (orb_weight * orb_M)

    # Calculate the angle, scale and displacement
    angle = np.arctan2(weighted_M[1, 0], weighted_M[0, 0])
    scale = np.sqrt(weighted_M[0, 0]**2 + weighted_M[1, 0]**2)
    displacement = np.array([weighted_M[0, 2], weighted_M[1, 2]])
    T = np.array([[np.cos(angle), -np.sin(angle), displacement[0]],
                    [np.sin(angle), np.cos(angle), displacement[1]],
                    [0, 0, 1]])
    
    return weighted_M, scale, angle, displacement, T

def get_weighted_M3(orb_disp, akaze_disp, orb_rot, akaze_rot, orb_n_matches, akaze_n_matches):
    # Calculate weights based on the number of matches
    total_matches = orb_n_matches + akaze_n_matches
    if total_matches == 0:
        return np.eye(3), 0, np.array([0, 0])  # Return identity matrix and zeros if no matches

    orb_weight = orb_n_matches / total_matches
    akaze_weight = akaze_n_matches / total_matches

    weighted_disp = (orb_weight * orb_disp) + (akaze_weight * akaze_disp)
    weighted_rot = (orb_weight * orb_rot) + (akaze_weight * akaze_rot)
    weighted_M = np.array([[np.cos(weighted_rot), -np.sin(weighted_rot), weighted_disp[0]],
                           [np.sin(weighted_rot), np.cos(weighted_rot), weighted_disp[1]],
                           [0, 0, 1]])
    
    return weighted_M, weighted_rot, weighted_disp

# **********************************************************************************************
# MAIN FUNCTION
def main(args=None):
    rclpy.init(args=args)
    logging.info(f"{Colors.RED}Here we go!!!{Colors.RESET}")  
    sar_gen = SAR('/UWBradar0/readings', # Left radar
                    '/UWBradar1/readings' # Right radar
                    )
    sar_gen.get_logger().info(f"{Colors.GREEN}{Colors.BOLD}Bringup the robot...{Colors.RESET}")
    try:
        rclpy.spin(sar_gen)
    except KeyboardInterrupt:
        sar_gen.get_logger().info(f"{Colors.RED}Keyboard Interrupt{Colors.RESET}")
        
    sar_gen.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except:
        logging.error('Error in the main function')
        rclpy.shutdown()
        pass
