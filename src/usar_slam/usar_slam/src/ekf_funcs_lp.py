# pylint: disable=C0103, C0116, W0611, C0302, C0301, C0303, C0114
import math
import time
import pdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class Colors:
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

# Odometry to control input
def odom2u(odom0, odom1):
    """ Convert odometry to control input."""
    # odom0: [x0, y0, th0]
    # odom1: [x1, y1, th1]

    x0 = odom0[0]
    y0 = odom0[1]
    th0 = odom0[2]

    x1 = odom1[0]
    y1 = odom1[1]
    th1 = odom1[2]

    # Control input
    rot1 = math.atan2(y1 - y0, x1 - x0) - th0
    tran = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    rot2 = th1 - th0 - rot1

    return [tran, rot1, rot2]

# Test continuity count statistic
def cont_count(bins, thresh):
    """ Test continuity count statistic."""
    n = len(bins)
    stat = sum(bins >= thresh) / n
    return stat

# Find maximum Q value
def find_maxQ(n_bins, N_bar):
    """ Find maximum Q value stat."""
    maxQ = N_bar * (n_bins - 1) + N_bar * (n_bins - 1) ** 2
    return maxQ

# Find Quadrat count statistic:: main function
def quadratC(inlierPts, x, y, m, fig):
    """ Find Quadrat count statistic."""
    # inlierPts: inlier points
    # x: x values of the line
    # y: y values of the line
    # m: number of quadrats
    # fig: plot the points, line, and quadrat midpoints
    
    if m == 0: # When line segment is less than the quadrat size, m can be zero
        m = 1  # set m to 1 to avoid division by zero, and to consider the whole line segment

    n_Pts = inlierPts.shape[0]  # number of points
    bins = np.zeros(m)  # initialize bins

    x_min, x_max = x[0], x[1]  # minimum and maximum x values of the line
    y_min, y_max = y[0], y[1]  # minimum and maximum y values of the line

    X_quad = np.linspace(x_min, x_max, 2 * m + 1)  # x values for quadrat midpoints
    Y_quad = np.linspace(y_min, y_max, 2 * m + 1)  # y values for quadrat midpoints

    X_quad_mid = X_quad[1::2]  # x values for quadrat midpoints excluding boundaries
    Y_quad_mid = Y_quad[1::2]  # y values for quadrat midpoints excluding boundaries
    quadMid = np.column_stack((X_quad_mid, Y_quad_mid))  # coordinates of quadrat midpoints

    for i in range(n_Pts):  # iterate over each point
        dist = np.zeros(m)  # initialize distance array
        for j in range(m):  # iterate over each quadrat midpoint
            P = np.vstack((inlierPts[i], quadMid[j]))  # create a line segment between the point and quadrat midpoint
            dist[j] = np.linalg.norm(P[0] - P[1])  # calculate the distance between the point and quadrat midpoint
        I = np.argmin(dist)  # find the index of the closest quadrat midpoint
        bins[I] += 1  # increment the count of the corresponding bin

    N_bar = n_Pts / m  # expected number of points per bin
    maxQ = find_maxQ(m, N_bar)  # maximum Q value
    thresh = 1  # threshold for continuity count
    stat = cont_count(bins, thresh)  # continuity count statistic
    Q = np.sum((bins - N_bar) ** 2) / N_bar  # Quadrat count statistic
    mod_bins = np.copy(bins)  # create a copy of bins
    mod_bins[mod_bins > N_bar] = N_bar  # limit the count of each bin to N_bar
    modQ = np.sum((mod_bins - N_bar) ** 2) / N_bar  # modified Quadrat count statistic

    if fig:  # if fig is True, plot the points, line, and quadrat midpoints
        plt.scatter(inlierPts[:, 0], inlierPts[:, 1], color='r')
        plt.plot(x, y, 'g-')
        plt.scatter(X_quad_mid, Y_quad_mid, marker="*")
        plt.show()

    return Q, modQ, maxQ, n_Pts, bins, stat, N_bar


# Find Quadrat count statistic:: secondary function
def quadratC2(bins):
    n_Pts = sum(bins) # number of points
    n_bins = len(bins) # number of bins

    N_bar = n_Pts / n_bins # expected number of points per bin

    maxQ = find_maxQ(n_bins, N_bar)  # maximum Q value

    thresh = 1  # threshold for continuity count
    stat = cont_count(bins, thresh)  # continuity count statistic

    Q = sum((bins - N_bar) ** 2) / N_bar  # Quadrat count statistic
    bins[bins > N_bar] = N_bar  # limit the count of each bin to N_bar
    modQ = sum((bins - N_bar) ** 2) / N_bar  # modified Quadrat count statistic

    return Q, modQ, maxQ, n_Pts, bins, stat, N_bar

# Find the consecutive elements in an array
def findConsec(arr, val):
    # Initialize variables
    consecutive_count = 0 # count of consecutive elements
    consecutive_elements = np.array([]) # consecutive elements
    max_count = 0 # maximum count of consecutive elements
    arr_L = len(arr) # length of the array

    # Loop through the array
    for i in range(arr_L):
        if arr[i] >= val:
            consecutive_count += 1
            consecutive_elements = np.append(consecutive_elements, arr[i])
            if consecutive_count > max_count:
                max_count = consecutive_count # update the maximum count
                max_cons_elements = np.copy(consecutive_elements)# update the maximum consecutive elements
                max_ind = i
        else:
            # Reset consecutive elements and count
            consecutive_count = 0
            consecutive_elements = np.array([])

    if False: # Whether to apply tail ends correction
        # Correction for tail ends observations: e.g. [2, 0, 1, 4, 5, 1, 9, 0, 6, 0, 7, 8]
        ind1 = max_ind - max_count - 1 # index of the element before the start of the consecutive elements
        ind2 = max_ind + 2 # index of the element after the end of the consecutive elements
        if ind1 > 0:
            if arr[ind1] > val:
                max_count += 2
                np.concatenate((max_cons_elements, arr[ind1: ind1 + 2]))
        if ind2 < len(arr):
            if arr[ind2] > val:
                max_count += 2
                np.concatenate((max_cons_elements, arr[ind2 - 1: ind2]))

    return max_count, max_cons_elements

# Dind perpendicular distance between a point and a line
def pt2line(lineParams, point):
    # lineParams = [m, c]
    m = lineParams[0]
    c = lineParams[1]

    x = point[0]
    y = point[1]

    D = abs(m * x - y + c) / math.sqrt(m ** 2 + 1)
    return D

# Convert m and c to line parameters
def mc2line(X, points, randPts, m, c, figID, isFig):
    isFig = False
    
    if isFig:
        X = [min(randPts[0]), max(randPts[0])]
        Y = [m * X[0] + c, m * X[1] + c]

        plt.figure(figID)
        plt.figure(figID).clear()
        plt.scatter(points[:, 0], points[:, 1], color='black')
        plt.scatter(randPts[0], randPts[1], color='red', marker='o', edgecolors='black')
        plt.plot(X, Y, color='blue', linewidth=2)
        plt.xlim(X)
        plt.ylim([-2500, 2500])
        plt.ylim([min(points[:, 1]), max(points[:, 1])])
        plt.axis('equal')
        plt.pause(0.1)
        plt.show()

#_______________________________________________________________________________________________________________________
#
# Point Extraction
#_______________________________________________________________________________________________________________________

# Filter points that are close to each other
def filtnearPts(P_LMs, N_LMs, pts):
    thresh = 300
    D = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=-1) # distance of the point to all other points
    n = N_LMs # number of points
    p_LMs = np.empty((0, 2))
    n_LMs = 0

    for i in range(n): # for each point
        d = D[i] # distance of the point to all other points   
        if np.all(d[d != 0] > thresh): # check if the distance of the point to all other points is greater than the threshold
            p_LMs = np.vstack((p_LMs, P_LMs[i])) # add the point to the list of points
            n_LMs += 1 # increment the number of points

    N_LMs = n_LMs
    return p_LMs, N_LMs

# Extract points from the observations
def pointExtract(points, N_LMs, params, odom_i, rob_obs_pose, fig):
    P_LMs = np.zeros((2))
    xymeans = np.zeros((2))
    if points.shape[0] >= params[0]: # more than 5 points, preferably min_samples=params[0]
        cluster1 = DBSCAN(eps=params[1], min_samples=params[0], algorithm='auto') # clustering algorithm
        clus_idx = cluster1.fit_predict(points) # cluster the points

        N = np.max(clus_idx) + 1 # number of clusters

        xymeans = np.zeros((N, 2))
        P_LMs = np.zeros((N, 2))
        # lengths = np.zeros((N, N_LMs[0]))

        for i in range(N): # for each cluster
            ind = np.where(clus_idx == i)[0] # get the indices of the points in the cluster
            pt = np.mean(points[ind], axis=0) # get the mean of the points in the cluster

            xymeans[N_LMs] = pt

            odompt = rob_obs_pose.flatten()
            X = np.vstack((pt, odompt[:2]))
            r = np.linalg.norm(X[0] - X[1])
            P_LMs[N_LMs - 1, 0] = r

            del_pt = pt - odompt[:2]
            th = np.arctan2(del_pt[1], del_pt[0]) - odompt[2]
            P_LMs[N_LMs - 1, 1] = th

            N_LMs += 1

        if N_LMs > 1:
            P_LMs, N_LMs = filtnearPts(P_LMs, N_LMs, xymeans)

        if fig:
            plt.figure()
            plt.scatter(odom_i[:, 0], odom_i[:, 1], marker='.', color='black', label='Odometry')
            plt.scatter(xymeans[:N_LMs[1], 0], xymeans[:N_LMs[1], 1], color='blue', label='Cluster center(s)')
            plt.legend()
            plt.xlabel('x [mm]')
            plt.ylabel('y [mm]')
            plt.axis('equal')
            plt.show()

    return N_LMs, P_LMs, xymeans

#_______________________________________________________________________________________________________________________
#
# Tags Extraction
#_______________________________________________________________________________________________________________________



def tagExtract(tag_xy, params, odom_i, rob_obs_pose, fig):
    min_samp = params[0]
    eps = params[1]
    min_n = 50

    P_LMs = np.zeros((2))
    xymeans = np.zeros((2))
    N_tags = 0
    if tag_xy.shape[0] >= min_n:
        cluster1 = DBSCAN(eps=eps, min_samples=min_samp, algorithm='auto')
        clus_idx = cluster1.fit_predict(tag_xy)

        N = np.max(clus_idx) + 1

        xymeans = np.empty([0, 2])
        P_LMs = np.empty([0, 2])

        for i in range(N):
            ind = np.where(clus_idx == i)[0]
            tag_data = tag_xy[ind]
            
            std_x, std_y = np.std(tag_data, axis=0)
            if std_x < 150 and std_y < 150:
                mean_x, mean_y = np.mean(tag_data, axis=0)

                # Create filter for points within 1 standard deviation in both x and y
                filtered_data = tag_data[
                    (tag_data[:, 0] >= mean_x - std_x) & (tag_data[:, 0] <= mean_x + std_x) &
                    (tag_data[:, 1] >= mean_y - std_y) & (tag_data[:, 1] <= mean_y + std_y)
                    ]
                
                if filtered_data.shape[0] >= min_samp:
                
                    pt = np.mean(filtered_data, axis=0)
                    xymeans = np.vstack((xymeans, pt))

                    odompt = rob_obs_pose.flatten()
                    X = np.vstack((pt, odompt[:2]))
                    r = np.linalg.norm(X[0] - X[1])

                    del_pt = pt - odompt[:2]
                    th = np.arctan2(del_pt[1], del_pt[0]) - odompt[2]
                    P_LMs = np.vstack((P_LMs, [r, th]))

                    N_tags += 1

        if N_tags > 1:
            N_tags = 0
            xymeans = np.empty([0, 2])
            P_LMs = np.empty([0, 2])
        # if N > 1:
        #     P_LMs, N = filtnearPts(P_LMs, N, xymeans)

        if fig:
            plt.figure()
            plt.scatter(odom_i[:, 0], odom_i[:, 1], marker='.', color='black', label='Odometry')
            plt.scatter(xymeans[:N, 0], xymeans[:N, 1], color='blue', label='Cluster center(s)')
            plt.legend()
            plt.xlabel('x [mm]')
            plt.ylabel('y [mm]')
            plt.axis('equal')
            plt.show()

    return N_tags, P_LMs, xymeans

def tagExtract3(tag_xy, params):
    min_samp = params[0] # min points in a cluster
    eps = params[1] # max distance between points in a cluster
    min_n = params[2] # min total points to consider

    P_LMs = np.zeros((2))
    N_tags = 0
    if tag_xy.shape[0] >= min_n:
        cluster1 = DBSCAN(eps=eps, min_samples=min_samp, algorithm='auto')
        clus_idx = cluster1.fit_predict(tag_xy)

        # # Find the index of the largest cluster
        # clus_ind = largest_cluster_index(clus_idx)

        N = np.max(clus_idx) + 1

        # xymeans = np.empty([0, 2])
        P_LMs = np.empty([0, 2])

        for i in range(N):
            ind = np.where(clus_idx == i)[0]
            tag_data = tag_xy[ind]
            
            # convert to polar coordinates
            tag_data = cart2pol(tag_data) 
            std_r, std_th = np.std(tag_data, axis=0)
            print('tag cluster i:', i, 'std:', [std_r, std_th], 'n_samples:', tag_data.shape[0])
            

            if std_r < 50:
                # When the angles are discontinuous in the range of -pi to pi
                if std_th > 0.5: # if the standard deviation of the angle is greater than 0.5 rad
                    mean_th, std_th = circular_mean_std(tag_data[:, 1])
                    if std_th < 0.03: # 0.03 rad = 1.7 degrees
                        filt_r, filt_th = circ_filter_within_std(tag_data[:, 1], tag_data[:, 0], mean_th, std_th)

                        if filt_th.shape[0] >= min_samp:
                            # print('n_samples:', filt_th.shape[0])

                            mean_th, _ = circular_mean_std(filt_th)
                            mean_r = np.mean(filt_r)
                            pt = [mean_r, mean_th]
                            P_LMs = np.vstack((P_LMs, pt))

                            # r = math.sqrt(pt[0]**2 + pt[1]**2)
                            # th = np.arctan2(pt[1], pt[0])

                            # P_LMs = np.vstack((P_LMs, [r, th]))

                            N_tags += 1
                            print('modified std_th: ', std_th)


                # Using the modified standard deviation
                elif std_th < 0.03: # 0.03 rad = 1.7 degrees
                    mean_r, mean_th = np.mean(tag_data, axis=0)

                    # Create filter for points within 1 standard deviation in both x and y
                    filtered_data = tag_data[
                        (tag_data[:, 0] >= mean_r - std_r) & (tag_data[:, 0] <= mean_r + std_r) &
                        (tag_data[:, 1] >= mean_th - std_th) & (tag_data[:, 1] <= mean_th + std_th)
                        ]
                    
                    if filtered_data.shape[0] >= min_samp:
                        # print('n_samples:', filtered_data.shape[0])
                    
                        pt = np.mean(filtered_data, axis=0)
                        P_LMs = np.vstack((P_LMs, pt))

                        # P_LMs = np.vstack((P_LMs, [r, th]))
                        
                        N_tags += 1
                        print('std_th: ', std_th)

            if N_tags > 1:
                N_tags = 0
                P_LMs = np.empty([0, 2])

            # if N > 1:
            #     P_LMs, N = filtnearPts(P_LMs, N, xymeans)

            # if fig:
            #     plt.figure()
            #     plt.scatter(odom_i[:, 0], odom_i[:, 1], marker='.', color='black', label='Odometry')
            #     plt.scatter(xymeans[:N, 0], xymeans[:N, 1], color='blue', label='Cluster center(s)')
            #     plt.legend()
            #     plt.xlabel('x [mm]')
            #     plt.ylabel('y [mm]')
            #     plt.axis('equal')
            #     plt.show()

    return N_tags, P_LMs

# layered filtering
def tagExtract2(tag_xy, params):
    min_samp = params[0] # min points in a cluster
    eps = params[1] # max distance between points in a cluster
    min_n = params[2] # min total points to consider

    P_LMs = np.zeros((2))
    N_tags = 0
    if tag_xy.shape[0] >= min_n:
        cluster1 = DBSCAN(eps=eps, min_samples=min_samp, algorithm='auto')
        clus_idx = cluster1.fit_predict(tag_xy)

        # # Find the index of the largest cluster
        # clus_ind = largest_cluster_index(clus_idx)

        N = np.max(clus_idx) + 1

        # xymeans = np.empty([0, 2])
        P_LMs = np.empty([0, 2])

        for i in range(N):
            ind = np.where(clus_idx == i)[0]
            tag_data = tag_xy[ind]
            
            # convert to polar coordinates
            tag_data = cart2pol(tag_data) 
            std_r, std_th = np.std(tag_data, axis=0)
            print('tag cluster i:', i, 'std:', [std_r, std_th], 'n_samples:', tag_data.shape[0])
            

            if std_r < 50:
                # When the angles are discontinuous in the range of -pi to pi
                if std_th > 0.5: # if the standard deviation of the angle is greater than 0.5 rad
                    mean_th, std_th = circular_mean_std(tag_data[:, 1])
                    if std_th < 0.03: # 0.03 rad = 1.7 degrees
                        filt_r, filt_th = circ_filter_within_std(tag_data[:, 1], tag_data[:, 0], mean_th, std_th)

                        if filt_th.shape[0] >= min_samp:
                            # print('n_samples:', filt_th.shape[0])

                            mean_th, _ = circular_mean_std(filt_th)
                            mean_r = np.mean(filt_r)
                            pt = [mean_r, mean_th]
                            P_LMs = np.vstack((P_LMs, pt))

                            # r = math.sqrt(pt[0]**2 + pt[1]**2)
                            # th = np.arctan2(pt[1], pt[0])

                            # P_LMs = np.vstack((P_LMs, [r, th]))

                            N_tags += 1
                            print('modified std_th: ', std_th)


                # Using the modified standard deviation
                elif std_th < 0.06: # 0.03 rad = 1.7 degrees
                    mean_r, mean_th = np.mean(tag_data, axis=0)

                    # Create filter for points within 1 standard deviation in both x and y
                    filtered_data = tag_data[
                        (tag_data[:, 0] >= mean_r - std_r) & (tag_data[:, 0] <= mean_r + std_r) &
                        (tag_data[:, 1] >= mean_th - std_th) & (tag_data[:, 1] <= mean_th + std_th)
                        ]
                    
                    # print('n_samples:', filtered_data.shape[0])
                    std_r, std_th = np.std(filtered_data, axis=0)
                    if std_th < 0.03: # 0.03 rad = 1.7 degrees
                        mean_r, mean_th = np.mean(filtered_data, axis=0)

                        # Create filter for points within 1 standard deviation in both x and y
                        filtered_data = filtered_data[
                            (filtered_data[:, 0] >= mean_r - std_r) & (filtered_data[:, 0] <= mean_r + std_r) &
                            (filtered_data[:, 1] >= mean_th - std_th) & (filtered_data[:, 1] <= mean_th + std_th)
                            ]
                        
                        if filtered_data.shape[0] >= min_samp:
                            pt = np.mean(filtered_data, axis=0)
                            P_LMs = np.vstack((P_LMs, pt))

                            # P_LMs = np.vstack((P_LMs, [r, th]))
                            
                            N_tags += 1
                            print('std_th: ', std_th)

            if N_tags > 1:
                N_tags = 0
                P_LMs = np.empty([0, 2])

            # if N > 1:
            #     P_LMs, N = filtnearPts(P_LMs, N, xymeans)

            # if fig:
            #     plt.figure()
            #     plt.scatter(odom_i[:, 0], odom_i[:, 1], marker='.', color='black', label='Odometry')
            #     plt.scatter(xymeans[:N, 0], xymeans[:N, 1], color='blue', label='Cluster center(s)')
            #     plt.legend()
            #     plt.xlabel('x [mm]')
            #     plt.ylabel('y [mm]')
            #     plt.axis('equal')
            #     plt.show()

    return N_tags, P_LMs

#_______________________________________________________________________________________________________________________
#
# FINAL Line and Point Extraction
#_______________________________________________________________________________________________________________________

def createObs(N_LMs_lhs_1, P_LMs_lhs_1, N_LMs_rhs_1, P_LMs_rhs_1):

    obs_Pts = np.zeros((10, 4)) # 10 points with 4 parameters each >> [x, y, idx, side = 1 for LHS and 2 for RHS]
    count_P = 0

    # LHS
    for i in range(N_LMs_lhs_1): # for each point landmark observation in REGION 1
        x1, y1 = P_LMs_lhs_1[i, :2]
        obs_Pts[count_P, :2] = [x1, y1]
        obs_Pts[count_P, 2] = count_P
        obs_Pts[count_P, 3] = 1 # side = 1 for LHS
        count_P += 1

    # RHS
    for i in range(N_LMs_rhs_1): # for each point landmark observation in REGION 1
        x1, y1 = P_LMs_rhs_1[i, :2]
        obs_Pts[count_P, :2] = [x1, y1]
        obs_Pts[count_P, 2] = count_P
        obs_Pts[count_P, 3] = 2
        count_P += 1

    return obs_Pts

# Convert r and th line parameters to m and c
def rth2mc(mu, nPtLMs, n):
    lin_i = nPtLMs * 2 + 3 + 2 * n # index of the line parameters
    
    line_r = mu[lin_i]
    line_th = mu[lin_i + 1]

    m = np.tan(np.pi / 2 + line_th)
    c = line_r / np.sin(line_th)

    return m, c

# Convert r and th line parameters to m and c
def rth2mc2(mu, r, th):
    psi = th + mu[2]

    x = r * math.cos(psi) + mu[0]
    y = r * math.sin(psi) + mu[1]

    line_r = math.sqrt(x ** 2 + y ** 2)
    line_th = math.atan2(y, x)

    m = math.tan(math.pi / 2 + line_th)
    c = line_r / math.sin(line_th)

    return m, c

# Convert r0 and alpha line parameters to m and c
def roalp2mc(mu, ro, alp):
    psi = alp + mu[2]

    x1 = ro * np.cos(psi) + mu[0]
    y1 = ro * np.sin(psi) + mu[1]

    m = np.tan(np.pi / 2 + psi)
    c = y1 - m * x1

    return m, c

# Visualize the line and point landmarks
def visPLs(N_LMs, L_LMs, P_LMs, odompt, ax_L, ax_P):
    # Lines
    for i in range(N_LMs[0]):
        l_x = L_LMs[i, 5:7]
        l_y = L_LMs[i, 7:9]
        # if L_LMs[i, 2] < 0:
        #     l_x = np.flip(l_x)
        ax_L[i].set_xdata(l_x)
        ax_L[i].set_ydata(l_y)

    # Points
    p_x = odompt[0] + P_LMs[:N_LMs[1], 0] * np.cos(P_LMs[:N_LMs[1], 1] + odompt[2])
    p_y = odompt[1] + P_LMs[:N_LMs[1], 0] * np.sin(P_LMs[:N_LMs[1], 1] + odompt[2])

    ax_P.set_offsets(np.column_stack((p_x, p_y)))

# Visualize the line and point landmarks
def visPLs2(N_LMs, L_LMs, P_LMs, ax_L, ax_P, fig):
    # Lines
    # L_LMs = [r, psi, m, c, x0, x1, y0, y1]
    for i in range(N_LMs[0]):
        l_x = L_LMs[i, 5:7]
        l_y = L_LMs[i, 7:9]

        # if L_LMs[i, 3] < 0:
        #     l_x = np.flip(l_x)
        ax_L.set_xdata(l_x)
        ax_L.set_ydata(l_y)
        # m = L_LMs[i, 2]
        # c = L_LMs[i, 3]
        # ax.set_ydata([m * l_x[0] + c, m * l_x[1] + c])
        fig.canvas.draw()

    if N_LMs[0] == 0:
        ax_L.set_xdata([])
        ax_L.set_ydata([])
        fig.canvas.draw()
    
    # Points
    for i in range(N_LMs[1]):
        ax_P.set_offsets(P_LMs[i, :2])
        fig.canvas.draw()
    
    if N_LMs[1] == 0:
        ax_P.set_offsets([ None, None])
        fig.canvas.draw()

#_______________________________________________________________________________________________________________________
#
# EKF SLAM
#_______________________________________________________________________________________________________________________

# Predict the state and covariance
def ekf_unkown_predict(mu, sig, u, R, F):
    # mu: mean of the state
    # sig: covariance of the state
    # u: control input
    # R: control noise

    # Get the control input
    tran = u[0]
    rot1 = u[1]
    rot2 = u[2]

    # Odometry model
    odo = np.array([[tran*math.cos(rot1 + mu[2])],
            [tran*math.sin(rot1 + mu[2])],
            [rot1 + rot2]])
    
    # Predicted state
    mu_bar = mu + np.matmul(np.transpose(F), odo)
    
    # Jacobian of the motion model
    g = np.array([[0, 0, -tran*math.sin(rot1 + mu[2])],
            [0, 0,  tran*math.cos(rot1 + mu[2])],
            [0, 0, 0]])
    
    G = np.eye(len(mu)) + np.matmul(np.matmul(np.transpose(F), g), F)

    # Predicted covariance
    sig_bar = np.matmul(np.matmul(G, sig), np.transpose(G)) + np.matmul(np.matmul(F.T, R), F)

    return mu_bar, sig_bar

# Subtract angles
def subtract_angles(angle1, angle2):
    # angles are in radians
    
    # Subtract the angles
    result = angle1 - angle2
    
    # Adjust the result to be within the range -π to π (or -180° to 180°)
    while result > math.pi:
        result -= 2 * math.pi
    while result < -math.pi:
        result += 2 * math.pi
    
    return result

# POINTS:: Correct the state and covariance 
def EKF_unknown_pts_obs_correction(k, mu_bar, sig_bar, exp_pt_landm, exp_tags, Q, z):
    dx = mu_bar[3 + 2 * k] - mu_bar[0]
    dy = mu_bar[4 + 2 * k] - mu_bar[1]
    del_ = [dx[0], dy[0]]

    q = np.dot(del_, del_)
    sq = np.sqrt(q)

    z0 = np.array([sq, np.arctan2(del_[1], del_[0]) - mu_bar[2][0]]).reshape(-1, 1)

    if z[1] > np.pi and z0[1] < 0:
        z0[1] = 2 * np.pi + z0[1]

    elif z0[1] > np.pi and z[1] < 0:
        z0[1] = z0[1] - 2 * np.pi

    elif z0[1] < -np.pi and z[1] > 0:
        z0[1] = z0[1] + 2 * np.pi
    
    elif z[1] < -np.pi and z0[1] > 0:
        z[1] = z[1] + 2 * np.pi

    del_z = z - z0

    F_xk = np.block([
        [np.eye(3), np.zeros((3, 2 * k)), np.zeros((3, 2)), np.zeros((3, 2 * exp_pt_landm - 2 * k - 2)), np.zeros((3, 2 * exp_tags))],
        [np.zeros((2, 3)), np.zeros((2, 2 * k)), np.eye(2), np.zeros((2, 2 * exp_pt_landm - 2 * k - 2)), np.zeros((2, 2 * exp_tags))]])

    H = (1 / q) * np.array([[-del_[0] * sq, -del_[1] * sq, 0, del_[0] * sq, del_[1] * sq],
                            [del_[1], -del_[0], -q, -del_[1], del_[0]]]) @ F_xk

    psi = H @ sig_bar @ H.T + Q
    pie = del_z.T @ np.linalg.inv(psi) @ del_z

    return pie, psi, H, z0

# TAGS::
def EKF_known_tag_obs_correction(k, mu_bar, sig_bar, exp_pt_landm, exp_tags, Q, z, alp_tag):
    dx = mu_bar[3 + 2 * exp_pt_landm + 2 * k] - mu_bar[0]
    dy = mu_bar[4 + 2 * exp_pt_landm + 2 * k] - mu_bar[1]
    del_ = [dx[0], dy[0]]

    q = np.dot(del_, del_)
    sq = np.sqrt(q)

    z0 = np.array([sq, np.arctan2(del_[1], del_[0]) - mu_bar[2][0]]).reshape(-1, 1)

    del_z = z - z0
    del_z[1] = subtract_angles(z[1], z0[1])

    F_xk = np.block([
        [np.eye(3), np.zeros((3, 2 * exp_pt_landm)), np.zeros((3, 2 * k)), np.zeros((3,2)), np.zeros((3, 2 * exp_tags - 2 - 2 * k))],
        [np.zeros((2, 3)), np.zeros((2, 2 * exp_pt_landm)), np.zeros((2, 2 * k)), np.eye(2), np.zeros((2, 2 * exp_tags - 2 - 2 * k))]])

    H = (1 / q) * np.array([[-del_[0] * sq, -del_[1] * sq, 0, del_[0] * sq, del_[1] * sq],
                            [del_[1], -del_[0], -q, -del_[1], del_[0]]]) @ F_xk

    psi = H @ sig_bar @ H.T + Q
    pie = del_z.T @ np.linalg.inv(psi) @ del_z

    if pie > alp_tag or abs(del_z[1]) > np.pi/6:
        print(f"{Colors.RED}REJECTED tag observation!! pie: {pie} or del_z[1]: abs({del_z[1]}) < {np.pi/6} {Colors.RESET}")
        print(" z[1]:", z[1])
        print("z0[1]:", z0[1])
        print("del_z[1]:", del_z[1])
        print("corrected del_z[1]:", subtract_angles(z[1], z0[1]))
        pie = False
    else:
        print("\n z[1]:", z[1])
        print("z0[1]:", z0[1])
        print("Del_z:", del_z)
        print("pie:", pie)

    return pie, psi, H, del_z

# EKF correction for point landmarks
def EKF_unknown_correction_P(iter, mu_bar, sig_bar, obs_pts, all_lhs_pts, all_rhs_pts, Q_pts, hist_i, exp_pt_landm, exp_tags, N_pt, mu_at_last_obs, alp_pt):
    obs_pts = np.squeeze(obs_pts)
    len_obs_pts, _ = obs_pts.shape

    pub_obs = []
    pub_ind = []
    
    # Point landmarks
    for j in range(len_obs_pts):
        if obs_pts[j, 0] != 0:
            r = obs_pts[j, 0]
            phi = obs_pts[j, 1]
            side = obs_pts[j, 3] # side = 1 for LHS and 2 for RHS

            z = np.array([r, phi]).reshape(-1, 1)

            # rel_meas = np.array([r * np.cos(phi + mu_bar[2]), r * np.sin(phi + mu_bar[2])]).reshape(-1, 1)

            cent = np.array([mu_bar[0] + r * np.cos(phi + mu_bar[2]), mu_bar[1] + r * np.sin(phi + mu_bar[2])])

            if side == 1: # LHS
                iso = chkIsolated(np.array(all_lhs_pts), cent, 150, 300, offset=1)
            elif side == 2: # RHS
                iso = chkIsolated(np.array(all_rhs_pts), cent, 150, 300, offset=1)

            if iso:
                mu_bar[1 + 2 * (N_pt + 1):3 + 2 * (N_pt + 1)] = cent

                pie = np.zeros(N_pt + 1)
                for k in range(N_pt):
                    pie[k], _, _, _ = EKF_unknown_pts_obs_correction(k, mu_bar, sig_bar, exp_pt_landm, exp_tags, Q_pts, z)
                pie[N_pt] = alp_pt
                print('\nPoints pie: ', pie)
                ind_j_pt = int(np.argmin(pie))
                print('Point:', ind_j_pt, "EKF Update obs ind:", ind_j_pt)
                # print(ind_j_pt)

                _, psi_j, H_j, z_j = EKF_unknown_pts_obs_correction(ind_j_pt, mu_bar, sig_bar, exp_pt_landm, exp_tags, Q_pts, z)

                N_pt = max(N_pt, ind_j_pt + 1)
                
                K = sig_bar @ H_j.T @ np.linalg.inv(psi_j)

                mu_bar = mu_bar + K @ (z - z_j)
                sig_bar = (np.eye(sig_bar.shape[0]) - K @ H_j) @ sig_bar
                
                mu_at_last_obs = mu_bar[:3] # mu at the last observation

                hist_i[ind_j_pt] = iter # update the history of the point landmark's observation index

                pub_obs.extend([r, phi]) # publish the observations
                pub_ind.append(ind_j_pt) # publish the index of the observation

                # mu_bar, sig_bar, N_pt, countarr, hist_i = modifyPtLMs(iter, hist_i, countarr, [exp_pt_landm, exp_line_landm], 1, ind_j_pt, mu_bar, sig_bar, N_pt)
        else:
            break

    return mu_bar, sig_bar, N_pt, pub_obs, pub_ind, hist_i, mu_at_last_obs

# EKF correction for tags/anchors
def FKF_known_correction_T(iter, mu_bar, sig_bar, N_tag_obs, tag_obs, tag_id, Q_tags, hist_i, exp_pt_landm, exp_tags, mu_at_last_obs, pub_tags, alp_tag):
    # print('Total tag_obs:', N_tag_obs)
    for i in range(N_tag_obs):
        obs = tag_obs[i]
        r = obs[0]
        phi = obs[1]
        z = np.array([r, phi]).reshape(-1, 1)

        pie, psi, H, del_z = EKF_known_tag_obs_correction(tag_id, mu_bar, sig_bar, exp_pt_landm, exp_tags, Q_tags, z, alp_tag)

        if pie:
            # print('Accepted tag obs_i:', i)
            # print('obs_i:', obs)
            # if tag_id == 2:
                # print('cur_obs:', z)
                # print('exp_obs:', z0)
                # print('del_obs:', z - z0)
            K = sig_bar @ H.T @ np.linalg.inv(psi)
            
            del_mu = K @ del_z
            mu_bar = mu_bar + del_mu
            # print("mu_update:", del_mu[0:3])
            sig_bar = (np.eye(sig_bar.shape[0]) - K @ H) @ sig_bar

            mu_at_last_obs = mu_bar[:3] # mu at the last observation

            hist_i[exp_pt_landm + tag_id] = iter # update the history of the tag's observation index
            pub_tags[2*tag_id:2*tag_id+2] = [r, phi] # publish the observations
            # print('Updated using Tag:', tag_id)

    return mu_bar, sig_bar, pub_tags, hist_i, mu_at_last_obs

# Modify the mu and sigma matrices
def modify_mu_sig(mu, Sig, ids, exp_LM):
    n_pos = 3
    n_pts = exp_LM[0]
    n_ptSq = n_pts * 2
    n_lin = exp_LM[1]
    n_lnSq = n_lin * 2

    pt_id = ids[0]
    pt_idSq = pt_id * 2
    ln_id = ids[1]
    ln_idSq = ln_id * 2

    if pt_id > 0:
        Sig[pt_idSq - 1 + n_pos: n_pos + n_ptSq - 1, :] = Sig[pt_idSq + 1 + n_pos: n_pos + n_ptSq, :]
        Sig[:, pt_idSq - 1 + n_pos: n_pos + n_ptSq - 1] = Sig[:, pt_idSq + 1 + n_pos: n_pos + n_ptSq]
        Sig[n_pos + n_ptSq - 1: n_pos + n_ptSq, :] = 0
        Sig[:, n_pos + n_ptSq - 1: n_pos + n_ptSq] = 0
        Sig[n_pos + n_ptSq - 1, n_pos + n_ptSq - 1] = 1e6
        Sig[n_pos + n_ptSq, n_pos + n_ptSq] = 1e6

        mu[pt_idSq - 1 + n_pos: n_pos + n_ptSq - 1] = mu[pt_idSq + 1 + n_pos: n_pos + n_ptSq]
        mu[n_pos + n_ptSq - 1: n_pos + n_ptSq] = 0

    if ln_id > 0:
        Sig[ln_idSq - 1 + n_pos + n_ptSq: n_pos + n_ptSq + n_lnSq - 1, :] = Sig[ln_idSq + 1 + n_pos + n_ptSq: n_pos + n_ptSq + n_lnSq, :]
        Sig[:, ln_idSq - 1 + n_pos + n_ptSq: n_pos + n_ptSq + n_lnSq - 1] = Sig[:, ln_idSq + 1 + n_pos + n_ptSq: n_pos + n_ptSq + n_lnSq]
        Sig[n_pos + n_ptSq + n_lnSq - 1: n_pos + n_ptSq + n_lnSq, :] = 0
        Sig[:, n_pos + n_ptSq + n_lnSq - 1: n_pos + n_ptSq + n_lnSq] = 0
        Sig[n_pos + n_ptSq + n_lnSq, n_pos + n_ptSq + n_lnSq] = 1e6
        Sig[n_pos + n_ptSq + n_lnSq - 1, n_pos + n_ptSq + n_lnSq - 1] = 1e6

        mu[ln_idSq - 1 + n_pos + n_ptSq: n_pos + n_ptSq + n_lnSq - 1] = mu[ln_idSq + 1 + n_pos + n_ptSq: n_pos + n_ptSq + n_lnSq]
        mu[n_pos + n_ptSq + n_lnSq - 1: n_pos + n_ptSq + n_lnSq] = 0

    return mu, Sig

def modifyPtLMs(i, hist_i, countarr, expN, val_pt, ind_pt, mu, Sig, N_pts):
    # Initialize variables
    if ind_pt > 0:
        delt_i = i - hist_i[ind_pt]
        if delt_i < 30 or hist_i[ind_pt] == 0:
            countarr[ind_pt] += 1
        elif delt_i > 200 and countarr[ind_pt] < val_pt:
            countarr[ind_pt] = 0
            mu, Sig = modify_mu_sig(mu, Sig, [ind_pt, 0], expN)
            N_pts -= 1
            i = 0
        hist_i[ind_pt] = i

    return mu, Sig, N_pts, countarr, hist_i

def modifyLinLMs(i, hist_i, countarr, expN, val_ln, ind_ln, mu, Sig, N_line, max_ind, z, visLine_x, visLine_y):
    if ind_ln > 0:
        index_ln = expN[0] + ind_ln

        delt_i = i - hist_i[index_ln]

        if delt_i < 100 or hist_i[index_ln] == 0 or countarr[index_ln] > 50:
            countarr[index_ln] += 1
        elif delt_i > 130 and countarr[index_ln] < val_ln:
            countarr[index_ln] = 0
            countarr[index_ln:-1] = countarr[index_ln + 1:]
            countarr[-1] = 0
            mu, Sig = modify_mu_sig(mu, Sig, [0, ind_ln], expN)
            visLine_x[ind_ln:-1, :] = visLine_x[ind_ln + 1:, :]
            visLine_x[-1, :] = 0
            visLine_y[ind_ln:-1, :] = visLine_y[ind_ln + 1:, :]
            visLine_y[-1, :] = 0
            N_line -= 1
            i = 0

        if max_ind == 2:
            for k in range(1, expN[0] + 1):
                if countarr[k] > 20:
                    m, c = rth2mc2(mu, z[0], z[1])
                    D = pt2line([m, c], mu[2*k + 2: 2*k + 3])
                    if D < 50:
                        mu, Sig = modify_mu_sig(mu, Sig, [0, ind_ln], expN)
                        N_line -= 1
                        visLine_x[ind_ln:-1, :] = visLine_x[ind_ln + 1:, :]
                        visLine_x[-1, :] = 0
                        visLine_y[ind_ln:-1, :] = visLine_y[ind_ln + 1:, :]
                        visLine_y[-1, :] = 0
                        i = 0
                        countarr[index_ln] = 0
                        countarr[index_ln:-1] = countarr[index_ln + 1:]
                        countarr[-1] = 0
                        break

                if countarr[k] == 0:
                    break

        hist_i[index_ln] = i

    return mu, Sig, N_line, countarr, hist_i, visLine_x, visLine_y

# Update the limits of the SLAM plot
def updateLimsX(k, visLine_x, new_x):
    if visLine_x[k, 0] == 0:
        visLine_x[k, 0] = new_x[0]
        visLine_x[k, 1] = new_x[1]
    # elif new_x[0] < new_x[1]:
    else:
        if visLine_x[k, 0] > new_x[0]:
            visLine_x[k, 0] = new_x[0]
        if visLine_x[k, 1] < new_x[1]:
            visLine_x[k, 1] = new_x[1]
    # else:  # new_x[0] > new_x[1]
    #     pdb.set_trace()
    #     if visLine_x[k, 0] > new_x[1]:
    #         visLine_x[k, 0] = new_x[1]
    #     if visLine_x[k, 1] < new_x[0]:
    #         visLine_x[k, 1] = new_x[0]
    return visLine_x
def updateLimsY(k, visLine_y, new_y):
    if visLine_y[k, 0] == 0:
        visLine_y[k, 0] = new_y[0]
        visLine_y[k, 1] = new_y[1]
    elif visLine_y[k, 0] < visLine_y[k, 1]:
        if visLine_y[k, 0] > new_y[0]:
            visLine_y[k, 0] = new_y[0]
        if visLine_y[k, 1] < new_y[1]:
            visLine_y[k, 1] = new_y[1]
    else:  # new_y[0] > new_y[1]
        if visLine_y[k, 0] < new_y[0]:
            visLine_y[k, 0] = new_y[0]
        if visLine_y[k, 1] > new_y[1]:
            visLine_y[k, 1] = new_y[1]
    return visLine_y

# Checks whether the center is isolated from the rest
def chkIsolated(points, center, thresh1, thresh2, offset=0):
    iso = False
    center = center.reshape(1, -1)
    # draws two circles with different radii and check the number of inliers
    # thresh1 and thresh2 are circle radii
    # if both inlier counts are same, the center is isolated
    L = points.shape[0] # number of points
    dist = np.zeros(L) # stores the distance values

    for p in range(L): # calculates the distance
        dist[p] = np.linalg.norm(points[p] - center)

    # pdb.set_trace()
    # if sum(dist < thresh1) == sum(dist < thresh2) > 0: # if the number of points are same inside both circles,
    #     iso = True # the point is isolated

    sum_thresh1 = sum(dist < thresh1)
    sum_thresh2 = sum(dist < thresh2)
    if sum_thresh1 > 0 and (sum_thresh2 - sum_thresh1) <= offset:
        iso = True # the point is isolated
        print('Isolated: ', sum(dist < thresh2), sum(dist < thresh1))

    return iso

# Find the squared distance between two points
def pt2pt_sq(pt1, pt2):
    return (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2

# Find the minimum value and its index
def find_min_index(lst):
    min_value = min(lst)
    min_index = lst.index(min_value) 
    return min_index, min_value  # returns the index and the value

def pts2mc(pt1, pt2):
    m = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    c = pt1[1] - m * pt1[0]
    return m, c

def rotate_points(points, theta_rad):
    # theta is in radians
    # points are numpy array of shape (n, 2)

    # Define rotation matrix
    rotation_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                                 [np.sin(theta_rad), np.cos(theta_rad)]])

    # Apply rotation
    rotated_points = np.dot(points, rotation_matrix) # NEED TO Transpose the matrix, here rotates (-theta) around z-axis

    return rotated_points

def project_point_to_line(P, A, B):
    x1, y1 = P
    x2, y2 = A
    x3, y3 = B

    # Vector AB
    ABx = x3 - x2
    ABy = y3 - y2

    # Vector AP
    APx = x1 - x2
    APy = y1 - y2

    # Dot product of AP and AB
    dot_product = APx * ABx + APy * ABy

    # Magnitude squared of AB
    magnitude_squared = ABx**2 + ABy**2

    # Projection scalar t
    t = dot_product / magnitude_squared

    # Projection point coordinates
    Px = x2 + t * ABx
    Py = y2 + t * ABy

    return [Px, Py]

def get_indexes_below_threshold(input_list, threshold):
    return [index for index, value in enumerate(input_list) if value < threshold]

def sort_two_lists(list1, list2):
    """
    Sorts list1 in ascending order and rearranges list2
    so that the corresponding elements follow the sorted order of list1.

    Parameters:
    list1 (list): The list to be sorted.
    list2 (list): The list to be rearranged.

    Returns:
    tuple: A tuple containing the sorted list1 and rearranged list2.
    """
    # Sort the combined list based on the first list's elements
    sorted_combined = sorted(zip(list1, list2), reverse=False) # reverse=True for descending order

    # Unzip the sorted combined list
    _, sorted_list2 = zip(*sorted_combined)

    # Convert the tuples back to lists
    return list(sorted_list2)

def cart2pol(coords):
    # coords is a 2D numpy array where each row is [x, y]
    x = coords[:, 0]
    y = coords[:, 1]
    
    r = np.sqrt(x**2 + y**2)  # radial distance
    theta = np.arctan2(y, x)  # angle in radians
    
    return np.vstack((r, theta)).T  # return as a 2D array [r, theta]

from collections import Counter

def largest_cluster_index(dbscan_labels):
    # Exclude noise points (label == -1)
    labels = dbscan_labels[dbscan_labels != -1]
    
    # Count the number of elements in each cluster
    label_counts = Counter(labels)
    
    # Find the cluster with the maximum number of elements
    largest_cluster_label = max(label_counts, key=label_counts.get)
    
    return largest_cluster_label

# Find mean and standard deviation of angles when angles are close to +/- pi
def circular_mean_std(angles):
    # Convert angles from radians to x and y components
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Compute the mean of x and y components
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Compute the resultant vector length R
    R = np.sqrt(mean_x**2 + mean_y**2)
    
    # Mean angle
    mean_angle = np.arctan2(mean_y, mean_x)
    
    # Circular standard deviation
    circular_std = np.sqrt(-2 * np.log(R))
    
    return mean_angle, circular_std

def circ_filter_within_std(angles, r, mean_angle, circular_std):
    # Compute the difference between each angle and the mean, accounting for wraparound
    angle_diffs = np.arctan2(np.sin(angles - mean_angle), np.cos(angles - mean_angle))
    
    # Filter angles that are within 1 standard deviation
    filtered_angles = angles[np.abs(angle_diffs) <= circular_std]
    filtered_r = r[np.abs(angle_diffs) <= circular_std]
    
    return filtered_r, filtered_angles