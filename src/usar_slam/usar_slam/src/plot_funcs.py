#In[0]
from matplotlib import pyplot as plt
import logging
import math
from matplotlib.animation import FuncAnimation
import pickle
import numpy as np

# plt.rcParams['backend'] = 'Qt5Agg'
logging.basicConfig(level=logging.INFO)  # Set the log level to DEBUG. By default, it is WARNING only
logger = logging.getLogger(__name__) # Create a logger

class Plotter:
    """Plotter class for visualizing the odometry-based raw observations and SLAM output"""
    def __init__(self):
        # Variables
        self.samp_size = 920/143
        n_samp = 1000
        self.dists = np.arange(0, n_samp) * self.samp_size
        max_dist = self.dists[-1]
        min_dist = self.dists[0]

        # Parameters for the SAR image
        res = 10 # Resolution in mm
        self.radar_range = [300, 3000] # Radar range in mm
        self.radar_azimuth = [math.radians(-30), math.radians(30)] # Radar azimuth in degrees
        self.pixels_x = int(2*self.radar_range[1]*math.sin(self.radar_azimuth[1])/res)
        self.pixels_y = int(self.radar_range[1]*math.cos(self.radar_azimuth[1])/res)
        print('resolution: ', res)
        print('pixels_x: ', self.pixels_x)
        print('pixels_y: ', self.pixels_y)
        self.radar1_img_size = (self.pixels_y, self.pixels_x)
        self.radar2_img_size = (self.pixels_y, self.pixels_x)
        self.radar3_img_size = (self.pixels_y, self.pixels_x)
        self.radar4_img_size = (self.pixels_y, self.pixels_x)

        self.radar_img_size = (self.pixels_y, self.pixels_x)

        # Initialize the raw images with zeros
        self.radar1_img = np.zeros(self.radar1_img_size, dtype=np.float32)
        self.radar2_img = np.zeros(self.radar2_img_size, dtype=np.float32)
        self.radar3_img = np.zeros(self.radar3_img_size, dtype=np.float32)
        self.radar4_img = np.zeros(self.radar4_img_size, dtype=np.float32)

        self.radar_img_mask = np.zeros(self.radar_img_size, dtype=np.int16)

        # assign dist indexes to the radar image pixels
        y = np.arange(res/2, res*self.pixels_y, res)
        if self.pixels_x % 2 == 0:
            x = np.arange(res/2, res*self.pixels_x/2, res)
        else:
            x = np.arange(res, res*(self.pixels_x +1)/2, res)

        tmp_radar_img_range = np.zeros((len(y), len(x)), dtype=np.float32)
        tmp_radar_img_azimuth = np.zeros((len(y), len(x)), dtype=np.float32)
        for i in range(len(y)):
            for j in range(len(x)):
                tmp_radar_img_range[i, j] = np.sqrt(x[j]**2 + y[i]**2)
                tmp_radar_img_azimuth[i, j] = np.arctan(x[j]/y[i])

        flipped_range = np.fliplr(tmp_radar_img_range)
        flipped_azimuth = np.fliplr(tmp_radar_img_azimuth) * -1

        if self.pixels_x % 2 == 0:
            self.radar_img_range = np.concatenate((flipped_range, tmp_radar_img_range), axis=1)
            self.radar_img_azimuth = np.concatenate((flipped_azimuth, tmp_radar_img_azimuth), axis=1)
        else:
            # print("flipped_range: ", flipped_range)
            # print("tmp_radar_img_range: ", tmp_radar_img_range)
            # print("reshaped y: ", np.reshape(y, (len(y), 1)))
            tmp_radar_img_range2 = np.concatenate((flipped_range, np.reshape(y, (len(y), 1))), axis=1)
            self.radar_img_range = np.concatenate((tmp_radar_img_range2, tmp_radar_img_range), axis=1)
            # print("radar_img_range: ", self.radar_img_range)
            tmp_radar_img_azimuth2 = np.concatenate((flipped_azimuth, np.zeros((len(y), 1))), axis=1)
            self.radar_img_azimuth = np.concatenate((tmp_radar_img_azimuth2, tmp_radar_img_azimuth), axis=1)
            # print("radar_img_azimuth: ", self.radar_img_azimuth)

        for i in range(self.pixels_y):
            for j in range(self.pixels_x):
                if self.radar_azimuth[0] <= self.radar_img_azimuth[i, j] <= self.radar_azimuth[1]:
                    if self.radar_img_range[i, j] >= self.radar_range[0] and self.radar_img_range[i, j] <= self.radar_range[1]:
                        near_ind = self.find_nearest_index(self.dists, self.radar_img_range[i, j])
                        self.radar_img_mask[i, j] = near_ind
        
        #mirrored: upside down
        self.radar_img_mask2 = np.flipud(self.radar_img_mask)

    #In[1]

        self.odom_hist_x = [0]
        self.odom_hist_y = [0]
        self.mu_hist_x = [0]
        self.mu_hist_y = [0]
        self.exp_tags = 0
        self.exp_pt_landm = 0
        self.th = [0, 90, 180, -90] # anchorangles in degrees. Clockwise A0, A1, A2, A3
        
        # In[1]
        # **********************************************************************************************
        # Raw plotter || UWB radar || LHS
        self.lhs_fig = plt.figure()
        self.ax_1 = self.lhs_fig.add_subplot(121) 
        self.ax_1.set_aspect('equal', adjustable='box')
        # plt.axis('equal')
        # self.ax.grid(linestyle="--", color='black', alpha=0.3)
        self.ax_1.set_xlabel('x [mm]')
        self.ax_1.set_ylabel('y [mm]')
        self.ax_1.set_title('Radar_1 observation')
        # self.ax_lhs.set_xlim(-100, 10000)
        # self.ax_lhs.set_ylim(-3000, 3000)
        # plt.rcParams['backend'] = 'Qt5Agg'

        self.ax_2 = self.lhs_fig.add_subplot(122) 
        self.ax_2.set_aspect('equal', adjustable='box')
        self.ax_2.set_xlabel('x [mm]')
        self.ax_2.set_ylabel('y [mm]')
        self.ax_2.set_title('Radar_2 observation')

        self.lhs_fig.canvas.draw()

        # **********************************************************************************************
        # Raw plotter || UWB radar || RHS
        self.rhs_fig = plt.figure()
        self.ax_3 = self.rhs_fig.add_subplot(121)
        self.ax_3.set_aspect('equal', adjustable='box')
        self.ax_3.set_xlabel('x [mm]')
        self.ax_3.set_ylabel('y [mm]')
        self.ax_3.set_title('Radar_3 observation')

        self.ax_4 = self.rhs_fig.add_subplot(122)
        self.ax_4.set_aspect('equal', adjustable='box')
        self.ax_4.set_xlabel('x [mm]')
        self.ax_4.set_ylabel('y [mm]')
        self.ax_4.set_title('Radar_4 observation')

        self.rhs_fig.canvas.draw()

        # **********************************************************************************************
        # Raw plotter || UWB SAR image
        self.sar_fig = plt.figure()
        self.ax_sar = self.sar_fig.add_subplot(111)
        self.ax_sar.set_aspect('equal', adjustable='box')
        # self.ax2.grid(linestyle="--", color='black', alpha=0.3)
        self.ax_sar.set_xlabel('x [mm]')
        self.ax_sar.set_ylabel('y [mm]')
        self.ax_sar.set_title('SAR image')
        # plt.rcParams['backend'] = 'Qt5Agg'
        # self.ax_sar.set_xlim(-10000, 10000)
        # self.ax_sar.set_ylim(-10000, 10000)

        self.sar_fig.canvas.draw()

        # **********************************************************************************************
        # Slam plotter
        self.slam_fig = plt.figure()
        self.slam_ax = self.slam_fig.add_subplot(111)
        self.slam_ax.set_aspect('equal', adjustable='box')
        # self.slam_ax.grid(linestyle="--", color='black', alpha=0.3)
        self.slam_ax.set_xlabel('x [mm]')
        self.slam_ax.set_ylabel('y [mm]')
        self.slam_ax.set_title('SLAM output')
        # plt.rcParams['backend'] = 'Qt5Agg'

        # Estimated path plot
        self.raw_odom, = self.slam_ax.plot([], [], linestyle='--', linewidth = 2, c='b', label='Odometry')
        self.est_path, = self.slam_ax.plot([], [], linewidth = 2, c='red', label='Estimated path')
        # Estimated point landmarks plot
        self.est_pt_lms = self.slam_ax.scatter([], [], s=60, c='magenta', marker='s', label='Estimated point landmarks')
        # Estimated tags plot
        self.est_tags = self.slam_ax.scatter([], [], s=60, c='green', marker='^', label='Estimated tags')

        self.slam_fig.canvas.draw()
        # self.fig3.canvas.flush_events()
        print('slam plotter flushed')

        # Legend
        self.slam_fig.legend(handles=[self.raw_odom, self.est_path, self.est_pt_lms, self.est_tags], loc='upper right')
        logging.info('SLAM plotter initialised...')
    
    def find_nearest_index(self, lst, number):
        # Use `min` with a custom key to find the index of the closest value
        nearest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - number))
        return nearest_index

    def terminate(self):
        plt.close('all')
        
    def call_back_raw_radar(self):
        while self.pipe_odoms.poll():
            odoms = self.pipe_odoms.recv()
            del_odom = self.pipe_del_odoms.recv()
            obs_lhs_R1 = self.pipe_obs_lhs_R1.recv()
            obs_rhs_R1 = self.pipe_obs_rhs_R1.recv()
            obs_lhs_R12 = self.pipe_obs_lhs_R12.recv()
            obs_rhs_R12 = self.pipe_obs_rhs_R12.recv()
            
            N_LMs_l_R1 = self.pipe_N_LMs_l_R1.recv()
            P_LMs_l_R1 = self.pipe_P_LMs_l_R1.recv()
            N_LMs_r_R1 = self.pipe_N_LMs_r_R1.recv()
            P_LMs_r_R1 = self.pipe_P_LMs_r_R1.recv()
            
            if obs_lhs_R1 is None:
                self.terminate()
            else:
                self.odom_hist_x.append(odoms[1][0])
                self.odom_hist_y.append(odoms[1][1])
                mod_odoms = odoms - del_odom

                self.ax.clear()
                
                self.ax.grid(linestyle="--", color='black', alpha=0.3)
                self.ax.scatter(obs_lhs_R12[:, 0], obs_lhs_R12[:, 1], s=30, c='g', marker='*', label='raw_observations R12')
                self.ax.scatter(obs_rhs_R12[:, 0], obs_rhs_R12[:, 1], s=30, c='g', marker='*', label='raw_observations R12')
                self.ax.scatter(obs_lhs_R1[:, 0], obs_lhs_R1[:, 1], s=30, c='b', marker='*', label='raw_observations R1')
                self.ax.scatter(obs_rhs_R1[:, 0], obs_rhs_R1[:, 1], s=30, c='b', marker='*', label='raw_observations R1')
                self.ax.plot(mod_odoms[:, 0], mod_odoms[:, 1], linewidth = 2, c='k', marker='o', label='Odometry')

                self.vis_p(N_LMs_l_R1, P_LMs_l_R1, 'purple')
                self.vis_p(N_LMs_r_R1, P_LMs_r_R1, 'purple')

                '''
                # Plot the odometry
                set_odom_dir(self.fig, self.odom_dir, self.odom1 - self.del_odom)
                '''
                # Adjust the axis limits
                self.ax.set_xlim(mod_odoms[0, 0]-3000, mod_odoms[0, 0]+3000)
                self.ax.set_ylim(mod_odoms[0, 1]-3000, mod_odoms[0, 1]+3000)

                self.fig.canvas.draw()
                # self.fig.canvas.flush_events()
    
    def callback_radar(self, radar_id, pipe_obs, ax, fig, radar_img, radar_img_obj):
        while pipe_obs.poll():
            obs = pipe_obs.recv()
            ax.set_title(f'Radar_{radar_id} observation: 1pix = 10mm')
            radar_img = obs[self.radar_img_mask]
            if hasattr(radar_img_obj):
                radar_img_obj.set_data(radar_img)   
            else:
                radar_img_obj = ax.imshow(radar_img, cmap='jet', animated=True)
            fig.canvas.draw_idle()

    def callback_radar1(self):
        while self.pipe_obs_R1.poll():
            obs_lhs_R1 = self.pipe_obs_R1.recv()

            self.ax_1.set_title('Radar_1 observation: 1pix = 10mm')

            self.radar1_img = obs_lhs_R1[self.radar_img_mask2] # flipped image mask
            if hasattr(self, 'radar1_img_obj'):
                self.radar1_img_obj.set_data(self.radar2_img)
            else:
                # If image hasn't been initialized, use imshow
                self.radar1_img_obj = self.ax_1.imshow(self.radar1_img, cmap='jet', animated=True)

            # Draw the updated figure
            self.lhs_fig.canvas.draw_idle()

    def callback_radar2(self):
        while self.pipe_obs_R2.poll():
            obs_lhs_R2 = self.pipe_obs_R2.recv()
            
            self.ax_2.set_title('Radar_2 observation: 1pix = 10mm')

            self.radar2_img = obs_lhs_R2[self.radar_img_mask2] # Flipped image mask
            if hasattr(self, 'radar2_img_obj'):
                self.radar2_img_obj.set_data(self.radar2_img)
            else:
                # If image hasn't been initialized, use imshow
                self.radar2_img_obj = self.ax_2.imshow(self.radar2_img, cmap='jet', animated=True)
                

            # Draw the updated figure
            self.lhs_fig.canvas.draw_idle()
            # self.lhs_fig.canvas.flush_events()

    def callback_radar3(self):
        while self.pipe_obs_R3.poll():
            obs_lhs_R3 = self.pipe_obs_R3.recv()

            self.ax_3.set_title('Radar_3 observation: 1pix = 10mm')

            self.radar3_img = obs_lhs_R3[self.radar_img_mask]
            if hasattr(self, 'radar3_img_obj'):
                self.radar3_img_obj.set_data(self.radar3_img)
            else:
                # If image hasn't been initialized, use imshow
                self.radar3_img_obj = self.ax_3.imshow(self.radar3_img, cmap='jet', animated=True)

            # Draw the updated figure
            self.rhs_fig.canvas.draw_idle()

    def callback_radar4(self):
        while self.pipe_obs_R4.poll():
            obs_lhs_R4 = self.pipe_obs_R4.recv()

            self.ax_4.set_title('Radar_4 observation: 1pix = 10mm')

            self.radar4_img = obs_lhs_R4[self.radar_img_mask]
            if hasattr(self, 'radar4_img_obj'):
                self.radar4_img_obj.set_data(self.radar4_img)
            else:
                # If image hasn't been initialized, use imshow
                self.radar4_img_obj = self.ax_4.imshow(self.radar4_img, cmap='jet', animated=True)

            # Draw the updated figure
            self.rhs_fig.canvas.draw_idle
    
    def call_back_raw_aoa(self):
        while self.pipe_obs_tag_1.poll():
            obs_tag_1 = self.pipe_obs_tag_1.recv()
            obs_tag_2 = self.pipe_obs_tag_2.recv()
            obs_tag_3 = self.pipe_obs_tag_3.recv()
            obs_tag_4 = self.pipe_obs_tag_4.recv()
            N_tag_1 = self.pipe_N_LMs_tag_1.recv()
            N_tag_2 = self.pipe_N_LMs_tag_2.recv()
            N_tag_3 = self.pipe_N_LMs_tag_3.recv()
            N_tag_4 = self.pipe_N_LMs_tag_4.recv()
            P_tag_1 = self.pipe_P_LMs_tag_1.recv()
            P_tag_2 = self.pipe_P_LMs_tag_2.recv()
            P_tag_3 = self.pipe_P_LMs_tag_3.recv()
            P_tag_4 = self.pipe_P_LMs_tag_4.recv()
            
            if obs_tag_1 is None:
                self.terminate()
                print('Raw AOA plotter terminated...')
            else:
                self.ax2.clear()
                self.ax2.grid(linestyle="--", color='black', alpha=0.3)
                self.ax2.scatter(0, 0, s=150, c='black', marker='>', label='Robot')
                self.ax2.plot([0, 0.1], [0, 0], color = 'black', linestyle='--')

                self.ax2.scatter(obs_tag_1[:, 0], obs_tag_1[:, 1], s=10, c='black', marker='*', label='Tag 1')
                self.ax2.scatter(obs_tag_2[:, 0], obs_tag_2[:, 1], s=10, c='brown', marker='*', label='Tag 2')
                self.ax2.scatter(obs_tag_3[:, 0], obs_tag_3[:, 1], s=10, c='darkgreen', marker='*', label='Tag 3')
                self.ax2.scatter(obs_tag_4[:, 0], obs_tag_4[:, 1], s=10, c='darkblue', marker='*', label='Tag 4')

                self.vis_tags(N_tag_1, P_tag_1, 'cyan')
                self.vis_tags(N_tag_2, P_tag_2, 'magenta')
                self.vis_tags(N_tag_3, P_tag_3, 'lime')
                self.vis_tags(N_tag_4, P_tag_4, 'blue')
                
                '''
                # Plot the odometry
                set_odom_dir(self.fig, self.odom_dir, self.odom1 - self.del_odom)
                '''
                # Adjust the axis limits
                self.fig2.canvas.draw()
                # self.fig2.canvas.flush_events()
                print('aoa plot flushed')
                
    def vis_p(self, N_LMs, P_LMs, color = 'red'):
        # Points
        for i in range(N_LMs):
            self.ax.scatter(P_LMs[i, 0], P_LMs[i, 1], c=color,  facecolors= 'none', marker= 'o')
            
        # self.fig.canvas.draw()
    def vis_tags(self, N_tags, tags, color = 'black'):
        # Tags
        for i in range(N_tags):
            X = tags[i, 0]*math.cos(tags[i, 1])
            Y = tags[i, 0]*math.sin(tags[i, 1])
            self.ax2.plot([0, X], [0, Y], color = color, linestyle='--')
            self.ax2.scatter(X, Y, c=color, marker='^', s=30)
            
        # self.fig.canvas.draw()
 
    def call_back_slam(self):
        # print('SLAM plot called...')
        while self.pipe_mu.poll():
            mu = self.pipe_mu.recv()
            N_pts = self.pipe_N_pts.recv()
            N_tags = self.pipe_N_tags.recv()
            self.mu_hist_x.append(mu[0][0])
            self.mu_hist_y.append(mu[1][0])
            
            if mu is None:
                self.terminate()
                print('SLAM plotter terminated...')
            else:
                self.slam_ax.clear()
                self.slam_ax.grid(linestyle="--", color='black', alpha=0.3)
                self.slam_ax.plot(self.mu_hist_x, self.mu_hist_y, linewidth = 2, c='r', label='Est path')
                self.slam_ax.scatter(mu[0], mu[1], s=30, c='r', marker='o')
                # self.slam_ax.plot()
                self.slam_ax.plot(self.odom_hist_x, self.odom_hist_y, linestyle="--", linewidth = 2, c='b', label='Odometry')
                # Plot the estimated landmarks
                # Points
                self.slam_ax.scatter(mu[3:3+2*N_pts:2], mu[4:3+2*N_pts:2], s=50, c='magenta', marker='s', label='Est Pt LMs')
                # Tags
                start_i = 3 + 2*self.exp_pt_landm 
                self.slam_ax.scatter(mu[start_i: start_i+2*N_tags:2], mu[start_i +1:start_i+2*N_tags:2], s=80, c='green', marker='^', label='Est Tags')
                
                self.fig3.canvas.draw()
                    
    def __call__(self,
                 pipe_obs_R1, pipe_obs_R2, pipe_obs_R3, pipe_obs_R4, 
                 ):
        
        # Raw plotter pipes
        self.pipe_obs_R1 = pipe_obs_R1
        self.pipe_obs_R2 = pipe_obs_R2
        self.pipe_obs_R3 = pipe_obs_R3
        self.pipe_obs_R4 = pipe_obs_R4
        
        # Start the raw radar plotter timer
        timer_raw = self.lhs_fig.canvas.new_timer(interval=0.001)
        timer_raw.add_callback(self.callback_radar(1, pipe_obs_R1, self.ax_1, self.lhs_fig, self.radar1_img, self.radar1_img_obj))
        timer_raw.start()
        logger.info('Raw RADAR_1 plotter timer started...')

        timer_raw = self.lhs_fig.canvas.new_timer(interval=0.001)
        timer_raw.add_callback(self.callback_radar(2, pipe_obs_R2, self.ax_2, self.lhs_fig, self.radar2_img, self.radar2_img_obj))
        timer_raw.start()
        logger.info('Raw RADAR_2 plotter timer started...')

        timer_raw = self.rhs_fig.canvas.new_timer(interval=0.001)
        timer_raw.add_callback(self.callback_radar(3, pipe_obs_R3, self.ax_3, self.rhs_fig, self.radar3_img, self.radar3_img_obj))
        timer_raw.start()
        logger.info('Raw RADAR_3 plotter timer started...')

        timer_raw = self.rhs_fig.canvas.new_timer(interval=0.001)
        timer_raw.add_callback(self.callback_radar(4, pipe_obs_R4, self.ax_4, self.rhs_fig, self.radar4_img, self.radar4_img_obj))
        timer_raw.start()
        logger.info('Raw RADAR_4 plotter timer started...')

        timer_p_comp = self.sar_fig.canvas.new_timer(interval=0.001)
        
        
        plt.show()
