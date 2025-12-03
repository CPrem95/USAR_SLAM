import numpy as np
import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pdb
from sar_slam.src import sar_funcs as sar
import matplotlib
import threading
import gc
import time
import multiprocessing as mp
import open3d as o3d

plt.ion()  # Enable interactive mode for live updates
"""
LOAD/ACQUIRE DATA
"""
data = scipy.io.loadmat('sar_memory.mat')
obs_radar_L = data['obs_radar_L']
obs_radar_R = data['obs_radar_R']
odoms = data['odom']

poses_optimized = []
with open('poses_optimized.txt', 'r') as f: # includes optimized poses
    for line in f:
        values = [float(x) for x in line.strip().split()]
        poses_optimized.append(values)
poses_optimized = np.array(poses_optimized)

# with open('poses_original.txt', 'r') as f:
#     poses_original = []
#     for line in f:
#         values = [float(x) for x in line.strip().split()]
#         poses_original.append(values)
# poses_original = np.array(poses_original)

with open('my_pose_data.txt', 'r') as f: # includes original poses
    poses_original = []
    for line in f:
        values = [x for x in line.strip().split()]
        values = [float(x) for x in values[1:]]
        poses_original.append(values)
    # poses_original.append([np.nan, np.nan, np.nan, np.nan])  # Append NaN row at the end
poses_original = np.array(poses_original)

with open('my_loop_data.txt', 'r') as f: # includes loop closures
    loops_optimized = []
    for line in f:
        values = [x for x in line.strip().split()]
        values = [int(x) for x in values[1:3]]
        loops_optimized.append(values)
    # loops_optimized.append([np.nan, np.nan, np.nan, np.nan])  # Append NaN row at the end

loops_xy_original = []
for i in range(len(loops_optimized)): # to plot the loop closures on the original poses
    loops_xy_original.append([poses_original[loops_optimized[i][0]][1:3], poses_original[loops_optimized[i][1]][1:3]])

loops_xy_optimized = []
for i in range(len(loops_optimized)): # to plot the loop closures on the optimized poses
    loops_xy_optimized.append([poses_optimized[loops_optimized[i][0]][1:3], poses_optimized[loops_optimized[i][1]][1:3]])

for i in range(len(odoms)): # find the end of odometry data
    if odoms[i][0] == -1.0:
        end_odom_index = i
        print("End of odometry data at index:", i)
        break
"""
PARAMETERS
"""
radar_r = 180
d = 210
mpp = 5e-3
mph = 6e-3
gamma = (np.pi - np.radians(30)) / 2
D_1 = []
D_2 = []
samp_size = 920 / 143
n_samp = 1500  # Number of samples
max_qsize = 1000
memory_size = 50000
winsize = 200  # window size for nodes      # 250   | 200
img_window_1 = 250  # < img_window_2  # 300   | 250
img_window_2 = 300  # < 2*winsize      # 450   | 300
loop_min_matches = 5
fig_update_freq = int(winsize / 10)  # Update the figure every 25 steps
half_fov = np.radians(30)  # Half field of view in radians
res = 5 # Resolution in mm
radar_range = [300, 2500] # Radar range in mm
loop_nn_k = 3 # Loop closure find nearest neighbors k
loop_nn_r = 4000 # Loop closure find nearest neighbors radius
loop_ignore_hist = 5 # Loop closure ignore history of recent # nodes ( >=5 to have enough neighbors)
ransac_thresh = 20.0 # RANSAC threshold
sar_area_x = 33000 # SAR area in mm # 25000
sar_area_y = 35000 # SAR area in mm # 15000 | 22000

sar_orig_x = 3000 # SAR origin in mm: 3000/res = 600  || origin at lower left corner (as usual)
sar_orig_y = 20000 # SAR origin in mm: 2500/res = 500 # 6000 | 13000
sar_end_y = sar_area_y - sar_orig_y

pixels_x = int(sar_area_x / res)
pixels_y = int(sar_area_y / res)

# Define the range for x and y
x = np.arange(-sar_orig_x + res/2, sar_area_x - sar_orig_x + res/2, res)
y = np.arange(sar_area_y - sar_orig_y - res/2, -sar_orig_y - res/2, -res)

# Create the meshgrid
img_dist_X, img_dist_Y = np.meshgrid(x, y)
radar_img_size = (pixels_y, pixels_x)
# Initialize the raw images with zeros
radar_img = np.zeros(radar_img_size, dtype=np.float32)
mask_img = radar_img.copy()
empty_img = radar_img.copy()

Fs = 23.328e9 # Sampling frequency
fc = 7.29e9
BW = 1.4e9
frac_bw = BW/fc
PRF = 14e6
VTX = 0.6
uwb_t, uwb_pulse = sar.generate_uwb_pulse(Fs, fc, frac_bw, PRF, VTX)

"""
ODOMETRY
GENERATE SAR
"""
show_loops = True

sar_fig = plt.figure()
sar_fig.set_size_inches(16, 12)
sar_ax = sar_fig.add_subplot(111)
radar_img_obj = sar_ax.imshow(np.zeros(radar_img_size), cmap='jet', animated=True)
traj_obj = sar_ax.plot([], [], color='white', linewidth=1, label='Trajectory')
traj_nodes_obj = sar_ax.scatter([], [], color='white', edgecolors='black', s=20, label='Trajectory Nodes')

sar_ax.set_title('SAR Image')
sar_ax.set_xlabel('X (mm)')
sar_ax.set_ylabel('Y (mm)')
# sar_ax.grid(True)

mask_fig = plt.figure()
mask_ax = mask_fig.add_subplot(111)
mask_img_obj = mask_ax.imshow(np.zeros(radar_img_size), cmap='gray', animated=True)
mask_ax.set_title('Mask Image')
mask_ax.set_xlabel('X (mm)')
mask_ax.set_ylabel('Y (mm)')
mask_ax.grid(True)

poses_fig = plt.figure()
poses_ax = poses_fig.add_subplot(111)
# print("poses_optimized.shape", poses_optimized.shape)
# print("poses_optimized_x:", poses_optimized[:, 1])
# print("poses_optimized_y:", poses_optimized[:, 2])
edges_ax_obj = poses_ax.plot(poses_optimized[:, 1]*1000, poses_optimized[:, 2]*1000, c='blue', label='Pose-to-Pose edges', linewidth=2, zorder=1)
null_loop_ax_obj = poses_ax.plot([], [], c='green', label='Loop Closure edges', linewidth=2, zorder=3)
poses_ax_obj = poses_ax.scatter(poses_optimized[:, 1]*1000, poses_optimized[:, 2]*1000, c='red', edgecolors='black', label='Optimized Poses', s=20, linewidths=0.5, zorder=2)
poses_ax.set_title('Optimized Poses')
poses_ax.set_xlabel('X (mm)')
poses_ax.set_ylabel('Y (mm)')
poses_ax.set_axisbelow(True)
poses_ax.grid(True)
poses_ax.set_aspect('equal', adjustable='box')
poses_ax.legend()
poses_ax.set_xlim(np.min(poses_optimized[:, 1])*1000 - 1000, np.max(poses_optimized[:, 1])*1000 + 1000)
poses_ax.set_ylim(np.min(poses_optimized[:, 2])*1000 - 1000, np.max(poses_optimized[:, 2])*1000 + 1000)
poses_fig.canvas.draw()
poses_fig.canvas.flush_events()

if show_loops:
    for p1, p2 in loops_xy_optimized:
        poses_ax.plot([p1[0]*1000, p2[0]*1000], [p1[1]*1000, p2[1]*1000], 'g-', linewidth=2, zorder=0)

orig_poses_fig = plt.figure()
poses_ax = orig_poses_fig.add_subplot(111)
# poses_original.append([np.nan, np.nan, np.nan, np.nan])  # Append NaN row at the end
# print("poses_original.shape", poses_original.shape)
# print("poses_original_x:", poses_original[:, 1])
# print("poses_original_y:", poses_original[:, 2])
edges_ax_obj = poses_ax.plot(poses_original[:, 1]*1000, poses_original[:, 2]*1000, c='blue', label='Pose-to-Pose edges', linewidth=2, zorder=1)
null_loop_ax_obj = poses_ax.plot([], [], c='green', label='Loop Closure edges', linewidth=2, zorder=3)
poses_ax_obj = poses_ax.scatter(poses_original[:, 1]*1000, poses_original[:, 2]*1000, c='red', edgecolors='black', label='Original Poses (nodes)', s=20, linewidths=0.5, zorder=2)
poses_ax.set_title('Original Poses')
poses_ax.set_xlabel('X (mm)')
poses_ax.set_ylabel('Y (mm)')
poses_ax.set_axisbelow(True)
poses_ax.grid(True)
poses_ax.set_aspect('equal', adjustable='box')
poses_ax.legend()
poses_ax.set_xlim(np.min(poses_original[:, 1])*1000 - 1000, np.max(poses_original[:, 1])*1000 + 1000)
poses_ax.set_ylim(np.min(poses_original[:, 2])*1000 - 1000, np.max(poses_original[:, 2])*1000 + 1000)
orig_poses_fig.canvas.draw()
orig_poses_fig.canvas.flush_events()

if show_loops:
    for p1, p2 in loops_xy_original:
        poses_ax.plot([p1[0]*1000, p2[0]*1000], [p1[1]*1000, p2[1]*1000], 'g-', linewidth=2, zorder=0)

# Save the current figure
poses_fig.savefig('optimized_poses.png', dpi=300)
orig_poses_fig.savefig('original_pose_graph.png', dpi=300)

pdb.set_trace()  # Debugging breakpoint

radar_img_obj.set_data(radar_img)
sar_ax.set_title('SAR Image (wheel odometry-based)')
sar_fig.canvas.draw()
sar_fig.canvas.flush_events()

sar_odoms = []
for idx, odom in enumerate(odoms):
    break
    sar_odoms.append((odom[0]/res + sar_orig_x/res, sar_area_y/res - sar_orig_y/res - odom[1]/res))
    if odom[0] == -1.0:
        abs_sar = np.abs(radar_img)
        positive_sar = radar_img + abs_sar
        positive_sar[positive_sar > 1.5] = 1.5
        positive_sar_blur = cv2.GaussianBlur(positive_sar, (0, 0), 2)
        plt.figure(figsize=(16, 12), dpi=200)
        plt.imshow(positive_sar_blur, cmap='jet', extent=[x[0], x[-1], y[-1], y[0]])
        plt.title('SAR Image (HD)')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.colorbar(label='Intensity')
        plt.tight_layout()
        plt.savefig('sar_image_hd_odom.png', dpi=300)
        plt.close()
        print("End of odometry data reached at index:", idx)
        scipy.io.savemat('results_odom_img.mat', {
            'radar_img': radar_img,
            'positive_sar_blur': positive_sar_blur
        })
        break
    obs_radar_1 = obs_radar_L[idx]
    obs_radar_2 = obs_radar_R[idx]
    def process_radar_1():
        global radar_img, mask_img
        radar_img, mask_img = sar.add_sar_radar_1_regen(
            odom, obs_radar_1, img_dist_X, img_dist_Y, radar_img, mask_img,
            res, half_fov, sar_orig_x, sar_end_y, radar_r, samp_size
        )

    def process_radar_2():
        global radar_img, mask_img
        radar_img, mask_img = sar.add_sar_radar_2_regen(
            odom, obs_radar_2, img_dist_X, img_dist_Y, radar_img, mask_img,
            res, half_fov, sar_orig_x, sar_end_y, radar_r, samp_size
        )

    t1 = threading.Thread(target=process_radar_1)
    t2 = threading.Thread(target=process_radar_2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print(f"Processed at index {idx}")
    if idx % 100 == 0:
        print(f"Processed {idx} out of {len(odoms)} odometry entries")
        # Update the plot
        abs_sar = np.abs(radar_img)
        positive_sar = radar_img + abs_sar
        positive_sar[positive_sar > 0.75] = 0.75
        positive_sar_blur = cv2.GaussianBlur(positive_sar, (0, 0), 2)
        radar_img_obj.set_data(positive_sar_blur)
        radar_img_obj.set_clim(vmin=np.min(positive_sar_blur), vmax=np.max(positive_sar_blur))
        print("min_pix:", np.min(positive_sar_blur), "max_pix:", np.max(positive_sar_blur))
        print("max_mask:", np.max(mask_img), "min_mask:", np.min(mask_img))

        traj_obj[0].set_data(np.array(sar_odoms).T)
        traj_nodes_x = poses_optimized[:idx + 1, 1]/res + sar_orig_x/res
        traj_nodes_y = sar_area_y/res - sar_orig_y/res - poses_optimized[:idx + 1, 2]/res
        traj_nodes_obj.set_offsets(sar_odoms[-1])

        sar_fig.canvas.draw()
        sar_fig.canvas.flush_events()

        mask_img_obj.set_data(mask_img)
        mask_img_obj.set_clim(vmin=np.min(mask_img), vmax=np.max(mask_img))
        mask_fig.canvas.draw()
        mask_fig.canvas.flush_events()

"""
OPTIMIZED POSES
GENERATE SAR
"""
# Create the meshgrid
img_dist_X, img_dist_Y = np.meshgrid(x, y)
radar_img_size = (pixels_y, pixels_x)
# Initialize the raw images with zeros
radar_img = np.zeros(radar_img_size, dtype=np.float32)
mask_img = radar_img.copy()
empty_img = radar_img.copy()

mem_odom = []
mem_odom_sar = []
radar_img_obj.set_data(radar_img)
sar_ax.set_title('SAR Image (optimized poses-based)')
sar_fig.canvas.draw()
sar_fig.canvas.flush_events()

# Process radar 1 and radar 2 in parallel threads
def process_radar_1():
    global radar_img, mask_img
    radar_img, mask_img = sar.add_sar_radar_1_regen(
    odom, obs_radar_1, img_dist_X, img_dist_Y, radar_img, mask_img,
    res, half_fov, sar_orig_x, sar_end_y, radar_r, samp_size
    )

def process_radar_2():
    global radar_img, mask_img
    radar_img, mask_img = sar.add_sar_radar_2_regen(
    odom, obs_radar_2, img_dist_X, img_dist_Y, radar_img, mask_img,
    res, half_fov, sar_orig_x, sar_end_y, radar_r, samp_size
    )

for idx, i in enumerate(poses_optimized):
    break
    odom_init = i[1:4]  # x, y, theta
    odom_init[0] = odom_init[0] * 1000
    odom_init[1] = odom_init[1] * 1000
    for j in range(200):
        index = idx * 200 + j
        print(f"Processed at index {index}")
        odom00 = odoms[index] 
        odom01 = odoms[index + 1]

        if odom00[0] == -1.0 or odom01[0] == -1.0:
            # abs_sar = np.abs(radar_img)
            # positive_sar = radar_img + abs_sar
            # positive_sar[positive_sar > 1.5] = 1.5
            # positive_sar_blur = cv2.GaussianBlur(positive_sar, (0, 0), 2)
            # plt.figure(figsize=(16, 12), dpi=200)
            # plt.imshow(positive_sar_blur, cmap='jet', extent=[x[0], x[-1], y[-1], y[0]])
            # plt.scatter(traj_nodes_x, traj_nodes_y, color='white', s=1, label='Trajectory Nodes')
            # plt.title('SAR Image (HD)')
            # plt.xlabel('X (mm)')
            # plt.ylabel('Y (mm)')
            # plt.colorbar(label='Intensity')
            # plt.tight_layout()
            # plt.savefig('sar_image_hd_optimized.png', dpi=300)
            # plt.close()
            # print("End of odometry data reached at index:", index)

            # Update the plot
            abs_sar = np.abs(radar_img)
            positive_sar = radar_img + abs_sar
            positive_sar[positive_sar > 1] = 1
            positive_sar_blur = cv2.GaussianBlur(positive_sar, (0, 0), 2)
            radar_img_obj.set_data(positive_sar_blur)
            radar_img_obj.set_clim(vmin=np.min(positive_sar_blur), vmax=np.max(positive_sar_blur))
            print("min_pix:", np.min(positive_sar_blur), "max_pix:", np.max(positive_sar_blur))
            print("max_mask:", np.max(mask_img), "min_mask:", np.min(mask_img))

            traj_obj[0].set_data(np.array(mem_odom_sar).T)
            traj_nodes_x = poses_optimized[:idx + 1, 1]/res + sar_orig_x/res
            traj_nodes_y = sar_area_y/res - sar_orig_y/res - poses_optimized[:idx + 1, 2]/res
            traj_nodes_obj.set_offsets(np.array([traj_nodes_x, traj_nodes_y]).T)

            sar_fig.canvas.draw()
            sar_fig.canvas.flush_events()
            sar_fig.savefig('sar_image_hd_optimized.png', dpi=300)

            scipy.io.savemat('results_optimized_img.mat', {
                'radar_img': radar_img,
                'positive_sar_blur': positive_sar_blur,
                'mem_odom': mem_odom,
                'mem_odom_sar': mem_odom_sar
            })
            break

        rot1 = np.arctan2(odom01[1] - odom00[1], odom01[0] - odom00[0]) - odom00[2]
        tran1 = np.hypot(odom01[0] - odom00[0], odom01[1] - odom00[1])
        rot2 = odom01[2] - odom00[2] - rot1

        dx = tran1 * np.cos(rot1 + odom_init[2])
        dy = tran1 * np.sin(rot1 + odom_init[2])
        dth = rot1 + rot2

        odom = [odom_init[0] + dx, odom_init[1] + dy, odom_init[2] + dth]
        odom_init = odom[:]
        obs_radar_1 = obs_radar_L[index]
        obs_radar_2 = obs_radar_R[index]
        
        t1 = threading.Thread(target=process_radar_1)
        t2 = threading.Thread(target=process_radar_2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        mem_odom.append(odom[:])
        mem_odom_sar.append([odom[0]/res + sar_orig_x/res, sar_area_y/res - sar_orig_y/res - odom[1]/res])

        if index % 100 == 0:
            continue
            print(f"\033[1;33mProcessed {idx} out of {len(odoms)} odometry entries\033[0m")
            # Update the plot
            abs_sar = np.abs(radar_img)
            positive_sar = radar_img + abs_sar
            positive_sar[positive_sar > 1] = 1
            positive_sar_blur = cv2.GaussianBlur(positive_sar, (0, 0), 2)
            radar_img_obj.set_data(positive_sar_blur)
            radar_img_obj.set_clim(vmin=np.min(positive_sar_blur), vmax=np.max(positive_sar_blur))
            print("min_pix:", np.min(positive_sar_blur), "max_pix:", np.max(positive_sar_blur))
            print("max_mask:", np.max(mask_img), "min_mask:", np.min(mask_img))

            traj_obj[0].set_data(np.array(mem_odom_sar).T)
            traj_nodes_x = poses_optimized[:idx + 1, 1]/res + sar_orig_x/res
            traj_nodes_y = sar_area_y/res - sar_orig_y/res - poses_optimized[:idx + 1, 2]/res
            traj_nodes_obj.set_offsets(np.array([traj_nodes_x, traj_nodes_y]).T)

            sar_fig.canvas.draw()
            sar_fig.canvas.flush_events()

            # poses_ax_obj.set_offsets(poses_optimized[:idx + 1, 1:3])
            # edges = np.array(poses_optimized[:idx + 1, 1:3])
            # if idx > 0:
            #     edges_ax_obj[0].set_data(edges[:idx, 0], edges[:idx, 1])
            # poses_ax.set_xlim(np.min(poses_optimized[:, 1]) - 1000, np.max(poses_optimized[:, 1]) + 1000)
            # poses_ax.set_ylim(np.min(poses_optimized[:, 2]) - 1000, np.max(poses_optimized[:, 2]) + 1000)
            # poses_ax.legend()            
            # poses_fig.canvas.draw()
            # poses_fig.canvas.flush_events()

            mask_img_obj.set_data(mask_img)
            mask_img_obj.set_clim(vmin=np.min(mask_img), vmax=np.max(mask_img))
            mask_fig.canvas.draw()
            mask_fig.canvas.flush_events()

mem_odom = []
mem_opt_odom = []
for idx, i in enumerate(poses_optimized):
    odom_init = i[1:4]  # x, y, theta
    odom_init[0] = odom_init[0] * 1000
    odom_init[1] = odom_init[1] * 1000
    mem_opt_odom.append(odom_init[0:2])
    for j in range(200):
        index = idx * 200 + j

        odom00 = odoms[index] 
        odom01 = odoms[index + 1]

        if odom00[0] == -1.0 or odom01[0] == -1.0:
            break

        rot1 = np.arctan2(odom01[1] - odom00[1], odom01[0] - odom00[0]) - odom00[2]
        tran1 = np.hypot(odom01[0] - odom00[0], odom01[1] - odom00[1])
        rot2 = odom01[2] - odom00[2] - rot1

        dx = tran1 * np.cos(rot1 + odom_init[2])
        dy = tran1 * np.sin(rot1 + odom_init[2])
        dth = rot1 + rot2

        odom = [odom_init[0] + dx, odom_init[1] + dy, odom_init[2] + dth]
        odom_init = odom[:]
        mem_odom.append(odom[:])

# Plot odom trajectory optimized
odom_traj_fig = plt.figure()
traj_ax = odom_traj_fig.add_subplot(111)
mem_odom_np = np.array(mem_odom)
mem_opt_odom_np = np.array(mem_opt_odom)
print("mem_odom_np.shape", mem_odom_np.shape)
traj_ax.scatter(mem_odom_np[1:end_odom_index, 0], mem_odom_np[1:end_odom_index, 1], label='Optimized Odometry')
for i in range(0, len(mem_odom_np), 100):
    direction_indicator = 10 * np.array([np.cos(mem_odom_np[i, 2]), np.sin(mem_odom_np[i, 2])])
    traj_ax.arrow(mem_odom_np[i, 0], mem_odom_np[i, 1], direction_indicator[0], direction_indicator[1], 
                  head_width=10, head_length=10, fc='red', ec='red', alpha=0.5)
traj_ax.set_xlabel('X (mm)')
traj_ax.set_ylabel('Y (mm)')
traj_ax.set_title('Odometry Trajectory')
traj_ax.legend()
traj_ax.grid(True)
traj_ax.set_aspect('equal', adjustable='box')
odom_traj_fig.canvas.draw()
odom_traj_fig.canvas.flush_events()

"""
ICP !!!
ground truth: SLAM Toolbox
"""

data = scipy.io.loadmat('slam_toolbox_nodes.mat')
nodes = data['nodes']
mem_nodes = []

len_nodes = len(nodes)
for i in range(1, len_nodes - 2):
    print(f"Node ID: {nodes[i][0]}, X: {nodes[i][1]}, Y: {nodes[i][2]}")
    mem_nodes.append([nodes[i][1], nodes[i][2]])

mem_nodes = np.array(mem_nodes)*1000
# Plot ground truth nodes
opt_traj_fig = plt.figure()
opt_traj_ax = opt_traj_fig.add_subplot(111)
opt_traj_ax.scatter(mem_nodes[:, 0], mem_nodes[:, 1], label='Ground Truth Nodes')

opt_traj_ax.set_xlabel('X (mm)')
opt_traj_ax.set_ylabel('Y (mm)')
opt_traj_ax.set_title('SLAM_toolbox Trajectory')
opt_traj_ax.legend()
opt_traj_ax.grid(True)
opt_traj_fig.canvas.draw()
opt_traj_fig.canvas.flush_events()
plt.show(block=True)  # Prevent the figure from closing immediately


data = scipy.io.loadmat('gt_data.mat')
gt_x = data['gt_x']
gt_y = data['gt_y']
mem_gt = np.array([gt_x[0], gt_y[0]]).T
print("mem_gt:", mem_gt)
print("mem_gt.shape", mem_gt.shape)

# mem_nodes <<< from the SLAM Toolbox
# mem_odom <<< from the optimized odometry

# Use Open3D ICP to align mem_odom (optimized odometry) to mem_nodes (ground truth nodes)
source_points = mem_odom_np[1:end_odom_index, :2]
# target_points = mem_nodes
target_points = mem_gt

# Convert to 3D by adding a zero z-coordinate
source_points_3d = np.c_[source_points, np.zeros(source_points.shape[0])]
target_points_3d = np.c_[target_points, np.zeros(target_points.shape[0])]

# Convert to Open3D point clouds
source_pcd = o3d.geometry.PointCloud()
target_pcd = o3d.geometry.PointCloud()
source_pcd.points = o3d.utility.Vector3dVector(source_points_3d)
target_pcd.points = o3d.utility.Vector3dVector(target_points_3d)

# ICP parameters
threshold = 500.0  # distance threshold in mm
trans_init = np.eye(4)

reg_p2p = o3d.pipelines.registration.registration_icp(
    source_pcd, target_pcd, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

rmse = reg_p2p.inlier_rmse
print("ICP RMSE:", rmse)

print("ICP Transformation Matrix:")
print(reg_p2p.transformation)

transformation_theta = np.arctan2(reg_p2p.transformation[1, 0], reg_p2p.transformation[0, 0])
print("ICP Rotation Angle (Theta) degrees:", np.degrees(transformation_theta))

# Transform source points
source_pcd.transform(reg_p2p.transformation)
aligned_source = np.asarray(source_pcd.points)

# Plot aligned trajectory
plt.figure()
plt.plot(target_points[:, 0], target_points[:, 1], label='Ground Truth Trajectory', c='blue', linewidth=1)
plt.plot(aligned_source[:, 0], aligned_source[:, 1], label='Aligned Optimized Poses', c='orange', linewidth=1)
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.title('ICP Alignment: Optimized Poses to Ground Truth')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show(block=True)


"""
PLOT non-opt mapping
"""
data = scipy.io.loadmat('results_odom_img.mat')
radar_img = data['radar_img']
positive_sar_blur = data['positive_sar_blur']
plt.figure(figsize=(16, 12), dpi=300)
plt.imshow(positive_sar_blur, cmap='jet', extent=[x[0], x[-1], y[-1], y[0]])
plt.plot(odoms[:end_odom_index, 0], odoms[:end_odom_index, 1], color='white', linewidth=1.5, label='Trajectory')
plt.title('SAR Image (HD) - Odometry-based')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.colorbar(label='Intensity')
plt.tight_layout()
plt.savefig('sar_image_hd_odom_final.png', dpi=300)
plt.close() 

"""
PLOT optimized mapping
"""
data = scipy.io.loadmat('results_optimized_img.mat')
radar_img = data['radar_img']
positive_sar_blur = data['positive_sar_blur']
plt.figure(figsize=(16, 12), dpi=300)
plt.imshow(positive_sar_blur, cmap='jet')
plt.plot(mem_odom_np[:, 0]/res + sar_orig_x/res, -mem_odom_np[:, 1]/res + sar_end_y/res, color='white', linewidth=1.5, label='Trajectory')
plt.scatter(mem_opt_odom_np[:, 0]/res + sar_orig_x/res, -mem_opt_odom_np[:, 1]/res + sar_end_y/res, color='white', edgecolors='black', s=40, label='Trajectory Nodes')
# plt.plot(mem_odom_np[:, 0]/res, mem_odom_np[:, 1]/res, color='white', linewidth=1, label='Trajectory')
plt.title('SAR Image (HD) - Optimized Poses-based')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.colorbar(label='Intensity')
plt.tight_layout()
plt.savefig('sar_image_hd_optimized_final.png', dpi=300)
plt.close()