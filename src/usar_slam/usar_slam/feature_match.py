from optparse import OptionParser
import rclpy
from rclpy.node import Node
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from std_msgs.msg import Float32MultiArray
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as rot
import logging
from scipy.io import savemat, loadmat
import cv2
import gc
import threading

'''Parallel processing'''
# from numba import njit, prange
# from joblib import Parallel, delayed
# import threading
import multiprocessing as mp
from multiprocessing import Queue
from geometry_msgs.msg import Pose2D

plt.ion()
# Define ANSI escape codes for colors
class Colors:
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

# **********************************************************************************************

class Matcher(Node):
    def __init__(self):
        super().__init__('fearure_match')

        # Parameters
        self.sar_img_names_1 = ['cropped_sar_1_0.mat', 'cropped_sar_1_1.mat', 'cropped_sar_1_2.mat', 'cropped_sar_1_3.mat', 'cropped_sar_1_4.mat',
                            'cropped_sar_1_5.mat', 'cropped_sar_1_6.mat', 'cropped_sar_1_7.mat', 'cropped_sar_1_8.mat', 'cropped_sar_1_9.mat',
                            'cropped_sar_1_10.mat']
        self.sar_img_names_2 = ['cropped_sar_2_0.mat', 'cropped_sar_2_1.mat', 'cropped_sar_2_2.mat', 'cropped_sar_2_3.mat', 'cropped_sar_2_4.mat',
                            'cropped_sar_2_5.mat', 'cropped_sar_2_6.mat', 'cropped_sar_2_7.mat', 'cropped_sar_2_8.mat', 'cropped_sar_2_9.mat',
                            'cropped_sar_2_10.mat']
        self.sar_imgs_1 = []
        self.sar_imgs_2 = []

        # Create a figure and axes for displaying images
        self.img_fig, self.axes = plt.subplots(2, 11, figsize=(10, 55))
        self.sar_ax1 = [self.axes[0, i] for i in range(11)]
        self.sar_ax2 = [self.axes[1, i] for i in range(11)]
        self.img_fig.canvas.set_window_title('SAR Images')
        # Create a figure and axes for displaying descriptors
        self.des_fig, self.des_axes = plt.subplots(2, 11, figsize=(10, 55))
        self.des_ax1 = [self.des_axes[0, i] for i in range(11)]
        self.des_ax2 = [self.des_axes[1, i] for i in range(11)]
        self.des_fig.canvas.set_window_title('SAR Image Descriptors')
        # Create a figure and axes for displaying matches
        self.match_fig, self.match_axes = plt.subplots(1, 3, figsize=(90, 30))
        self.match_axes = [self.match_axes[i] for i in range(3)]
        self.match_fig.canvas.set_window_title('Matches')
        # Set titles for match subplots
        self.match_axes[0].set_title("SIFT Matches", fontsize=15)
        self.match_axes[1].set_title("SURF Matches", fontsize=15)
        self.match_axes[2].set_title("ORB Matches", fontsize=15)

        # Create a figure and axes for displaying filtered matches
        self.filter_fig, self.filter_axes = plt.subplots(1, 3, figsize=(90, 30))
        self.filter_axes = [self.filter_axes[i] for i in range(3)]
        self.filter_fig.canvas.set_window_title('Filtered Matches')
        # Set titles for filtered match subplots
        self.filter_axes[0].set_title("Filtered SIFT Matches", fontsize=15)
        self.filter_axes[1].set_title("Filtered SURF Matches", fontsize=15)
        self.filter_axes[2].set_title("Filtered ORB Matches", fontsize=15)

        # Initialize lists to store keypoints and descriptors    
        self.all_sift_kp1 = []
        self.all_sift_des1 = []
        self.all_sift_kp2 = []
        self.all_sift_des2 = []
        self.all_orb_kp1 = []
        self.all_orb_des1 = []
        self.all_orb_kp2 = []
        self.all_orb_des2 = []
        self.all_surf_kp1 = []
        self.all_surf_des1 = []
        self.all_surf_kp2 = []
        self.all_surf_des2 = []

        # Initialize lists to store matches
        self.good_sift = []
        self.good_surf = []
        self.good_orb = []

        self.run()
    
    def SIFT_extract_and_display_keypoints(self, cropped_sar_1, cropped_sar_2, ax1, ax2, fig):
        # Initialize SIFT detector
        sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.01, edgeThreshold=10, sigma=1.6)

        # Detect keypoints and compute descriptors for cropped_sar_1
        keypoints_1, descriptors_1 = sift.detectAndCompute(cropped_sar_1, None)

        # Detect keypoints and compute descriptors for cropped_sar_2
        keypoints_2, descriptors_2 = sift.detectAndCompute(cropped_sar_2, None)

        return keypoints_1, descriptors_1, keypoints_2, descriptors_2
    
    def ORB_extract_and_display_keypoints(self, cropped_sar_1, cropped_sar_2, ax1, ax2, fig):
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=5000)

        # Detect keypoints and compute descriptors for cropped_sar_1
        keypoints_1, descriptors_1 = orb.detectAndCompute(cropped_sar_1, None)

        # Detect keypoints and compute descriptors for cropped_sar_2
        keypoints_2, descriptors_2 = orb.detectAndCompute(cropped_sar_2, None)

        return keypoints_1, descriptors_1, keypoints_2, descriptors_2
    
    def SURF_extract_and_display_keypoints(self, cropped_sar_1, cropped_sar_2, ax1, ax2, fig):
        # Initialize SURF detector
        try:
            surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400, nOctaves=4, nOctaveLayers=2, extended=False, upright=False)
        except AttributeError:
            raise ImportError("SURF is not available. Please install opencv-contrib-python.")

        # Detect keypoints and compute descriptors for cropped_sar_1
        keypoints_1, descriptors_1 = surf.detectAndCompute(cropped_sar_1, None)

        # Detect keypoints and compute descriptors for cropped_sar_2
        keypoints_2, descriptors_2 = surf.detectAndCompute(cropped_sar_2, None)

        return keypoints_1, descriptors_1, keypoints_2, descriptors_2
    
    def load_sar_images(self):
        for i in range(len(self.sar_img_names_1)):
            # Load the SAR images
            sar_img_1 = loadmat(self.sar_img_names_1[i])['cropped_sar_1']  # Replace 'data' with the actual key in the .mat file
            sar_img_2 = loadmat(self.sar_img_names_2[i])['cropped_sar_2']  # Replace 'data' with the actual key in the .mat file
            sar_img_1 = cv2.normalize(sar_img_1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            sar_img_2 = cv2.normalize(sar_img_2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Append the images to the lists
            self.sar_imgs_1.append(sar_img_1)
            self.sar_imgs_2.append(sar_img_2)

            # Display the loaded SAR images
            self.sar_ax1[i].imshow(sar_img_1, cmap='jet', animated=True)
            self.sar_ax2[i].imshow(sar_img_2, cmap='jet', animated=True)
            self.img_fig.canvas.draw()
            self.img_fig.canvas.flush_events()

    def gen_kps_and_descriptors(self):
        for i in range(len(self.sar_img_names_1)):
            # Retrieve the SAR images
            sar_img_1 = self.sar_imgs_1[i]
            sar_img_2 = self.sar_imgs_2[i]

            # Extract and display keypoints using SIFT
            sift_kp1, sift_des1, sift_kp2, sift_des2 = self.SIFT_extract_and_display_keypoints(sar_img_1, sar_img_2, self.des_ax1[i], self.des_ax2[i], self.des_fig)
            self.all_sift_kp1.append(sift_kp1)
            self.all_sift_des1.append(sift_des1)
            self.all_sift_kp2.append(sift_kp2)
            self.all_sift_des2.append(sift_des2)

            # Extract and display keypoints using ORB
            orb_kp1, orb_des1, orb_kp2, orb_des2 = self.ORB_extract_and_display_keypoints(sar_img_1, sar_img_2, self.des_ax1[i], self.des_ax2[i], self.des_fig)
            self.all_orb_kp1.append(orb_kp1)
            self.all_orb_des1.append(orb_des1)
            self.all_orb_kp2.append(orb_kp2)
            self.all_orb_des2.append(orb_des2)

            # Extract and display keypoints using SURF
            surf_kp1, surf_des1, surf_kp2, surf_des2 = self.SURF_extract_and_display_keypoints(sar_img_1, sar_img_2, self.des_ax1[i], self.des_ax2[i], self.des_fig)
            self.all_surf_kp1.append(surf_kp1)
            self.all_surf_des1.append(surf_des1)
            self.all_surf_kp2.append(surf_kp2)
            self.all_surf_des2.append(surf_des2)

        return True
    
    # Match between images of region 1 and region 2
    def kp_matcher_1(self, img_ind_1, img_ind_2, type='sift', plot_data=True):
        img1 = self.sar_imgs_1[img_ind_1]
        img2 = self.sar_imgs_2[img_ind_2]
        if type == 'sift':
            kp1 = self.all_sift_kp1[img_ind_1]
            des1 = self.all_sift_des1[img_ind_1]
            kp2 = self.all_sift_kp2[img_ind_2]
            des2 = self.all_sift_des2[img_ind_2]
             
            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1,des2,k=2)
            # Apply ratio test
            self.good_sift = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    self.good_sift.append([m])

            if plot_data:
                # cv.drawMatchesKnn expects list of lists as matches.
                img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,self.good_sift,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                self.match_axes[0].imshow(img3)
                self.match_fig.canvas.draw()
                self.match_fig.canvas.flush_events()

        elif type == 'surf':
            kp1 = self.all_surf_kp1[img_ind_1]
            des1 = self.all_surf_des1[img_ind_1]
            kp2 = self.all_surf_kp2[img_ind_2]
            des2 = self.all_surf_des2[img_ind_2]

            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1,des2,k=2)
            # Apply ratio test
            self.good_surf = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    self.good_surf.append([m])

            if plot_data:
                # cv.drawMatchesKnn expects list of lists as matches.
                img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,self.good_surf,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                self.match_axes[1].imshow(img3)
                self.match_fig.canvas.draw()
                self.match_fig.canvas.flush_events()

        elif type == 'orb':
            kp1 = self.all_orb_kp1[img_ind_1]
            des1 = self.all_orb_des1[img_ind_1]
            kp2 = self.all_orb_kp2[img_ind_2]
            des2 = self.all_orb_des2[img_ind_2]

            # create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # Match descriptors.
            matches = bf.match(des1,des2)
            # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)

            if plot_data:
                # Draw first 10 matches.
                img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                self.match_axes[2].imshow(img3)
                self.match_fig.canvas.draw()
                self.match_fig.canvas.flush_events()
  
    # Match in the images of same region 
    def kp_matcher_2(self, img_ind_1, img_ind_2, type='sift', region=1, plot_data=True):
        if region == 1:
            # Load the SAR images
            img1 = self.sar_imgs_1[img_ind_1]
            img2 = self.sar_imgs_1[img_ind_2]
            if type == 'sift':
                kp1 = self.all_sift_kp1[img_ind_1]
                des1 = self.all_sift_des1[img_ind_1]
                kp2 = self.all_sift_kp1[img_ind_2]
                des2 = self.all_sift_des1[img_ind_2]

                # BFMatcher with default params
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1,des2,k=2)
                # Apply ratio test
                self.good_sift = []
                for m,n in matches:
                    if m.distance < 0.75*n.distance:
                        self.good_sift.append([m])

                if plot_data:
                    # cv.drawMatchesKnn expects list of lists as matches.
                    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,self.good_sift,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    self.match_axes[0].imshow(img3)
                    self.match_fig.canvas.draw()
                    self.match_fig.canvas.flush_events()

            elif type == 'surf':
                kp1 = self.all_surf_kp1[img_ind_1]
                des1 = self.all_surf_des1[img_ind_1]
                kp2 = self.all_surf_kp1[img_ind_2]
                des2 = self.all_surf_des1[img_ind_2]

                # BFMatcher with default params
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1,des2,k=2)
                # Apply ratio test
                self.good_surf = []
                for m,n in matches:
                    if m.distance < 0.75*n.distance:
                        self.good_surf.append([m])

                if plot_data:
                    # cv.drawMatchesKnn expects list of lists as matches.
                    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,self.good_surf,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    self.match_axes[1].imshow(img3)
                    self.match_fig.canvas.draw()
                    self.match_fig.canvas.flush_events()

            elif type == 'orb':
                kp1 = self.all_orb_kp1[img_ind_1]
                des1 = self.all_orb_des1[img_ind_1]
                kp2 = self.all_orb_kp1[img_ind_2]
                des2 = self.all_orb_des1[img_ind_2]

                # create BFMatcher object
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                # Match descriptors.
                matches = bf.match(des1,des2)
                # Sort them in the order of their distance.
                matches = sorted(matches, key = lambda x:x.distance)

                # Apply ratio test
                for m,n in matches:
                    if m.distance < 0.75*n.distance:
                        self.good_orb.append([m])

                if plot_data:
                    # Draw first 10 matches.
                    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    self.match_axes[2].imshow(img3)
                    self.match_fig.canvas.draw()
                    self.match_fig.canvas.flush_events()
            
        elif region == 2:
            # Load the SAR images
            img1 = self.sar_imgs_2[img_ind_1]
            img2 = self.sar_imgs_2[img_ind_2]
            if type == 'sift':
                kp1 = self.all_sift_kp2[img_ind_1]
                des1 = self.all_sift_des2[img_ind_1]
                kp2 = self.all_sift_kp2[img_ind_2]
                des2 = self.all_sift_des2[img_ind_2]

                # BFMatcher with default params
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1,des2,k=2)
                # Apply ratio test
                for m,n in matches:
                    if m.distance < 0.9*n.distance:
                        self.good_sift.append([m])

                if plot_data:
                    # cv.drawMatchesKnn expects list of lists as matches.
                    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,self.good_sift,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    self.match_axes[0].imshow(img3)
                    self.match_fig.canvas.draw()
                    self.match_fig.canvas.flush_events()

            elif type == 'surf':
                kp1 = self.all_surf_kp2[img_ind_1]
                des1 = self.all_surf_des2[img_ind_1]
                kp2 = self.all_surf_kp2[img_ind_2]
                des2 = self.all_surf_des2[img_ind_2]

                # BFMatcher with default params
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1,des2,k=2)
                # Apply ratio test
                self.good_surf = []
                for m,n in matches:
                    if m.distance < 0.9*n.distance:
                        self.good_surf.append([m])

                if plot_data:
                    # cv.drawMatchesKnn expects list of lists as matches.
                    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,self.good_surf,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    self.match_axes[1].imshow(img3)
                    self.match_fig.canvas.draw()
                    self.match_fig.canvas.flush_events()

            elif type == 'orb':
                kp1 = self.all_orb_kp2[img_ind_1]
                des1 = self.all_orb_des2[img_ind_1]
                kp2 = self.all_orb_kp2[img_ind_2]
                des2 = self.all_orb_des2[img_ind_2]


                """ # Default ORB code from OpenCV
                # create BFMatcher object
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                # Match descriptors.
                matches = bf.match(des1, des2)

                # Sort them in the order of their distance.
                matches = sorted(matches, key = lambda x:x.distance)
                
                if plot_data:
                    # Draw first 10 matches.
                    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    self.match_axes[2].imshow(img3)
                    self.match_fig.canvas.draw()
                    self.match_fig.canvas.flush_events()
                """
                bf = cv2.BFMatcher()
                matches2 = bf.knnMatch(des1, des2, k=2)
                # Apply ratio test
                for m,n in matches2:
                    if m.distance < 0.9*n.distance:
                        self.good_orb.append([m])
                
                if plot_data:
                    # cv.drawMatchesKnn expects list of lists as matches.
                    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,self.good_orb,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    self.match_axes[2].imshow(img3)
                    self.match_fig.canvas.draw()
                    self.match_fig.canvas.flush_events()
    
    def ransac_filter(self, img_ind_1, img_ind_2, type = 'sift', region=1, ransac_threshold = 25.0, plot_data=True):
        # Load the SAR images
        if region == 1:
            img1 = self.sar_imgs_1[img_ind_1]
            img2 = self.sar_imgs_1[img_ind_2]
            if type == 'sift':
                kp1 = self.all_sift_kp1[img_ind_1]
                kp2 = self.all_sift_kp1[img_ind_2]
            elif type == 'surf':
                kp1 = self.all_surf_kp1[img_ind_1]
                kp2 = self.all_surf_kp1[img_ind_2]
            elif type == 'orb':
                kp1 = self.all_orb_kp1[img_ind_1]
                kp2 = self.all_orb_kp1[img_ind_2]
            else:
                raise ValueError("Invalid type. Choose 'sift', 'surf', or 'orb'.")

        elif region == 2:
            img1 = self.sar_imgs_2[img_ind_1]
            img2 = self.sar_imgs_2[img_ind_2]
            if type == 'sift':
                kp1 = self.all_sift_kp2[img_ind_1]
                kp2 = self.all_sift_kp2[img_ind_2]
            elif type == 'surf':
                kp1 = self.all_surf_kp2[img_ind_1]
                kp2 = self.all_surf_kp2[img_ind_2]
            elif type == 'orb':
                kp1 = self.all_orb_kp2[img_ind_1]
                kp2 = self.all_orb_kp2[img_ind_2]
            else:
                raise ValueError("Invalid type. Choose 'sift', 'surf', or 'orb'.")
        else:
            raise ValueError("Invalid region. Choose 1 or 2.")
        
        # Use the good matches from the previous step
        matches = {
            'sift': self.good_sift,
            'surf': self.good_surf,
            'orb': self.good_orb
        }.get(type, [])   

        print(f"Number of total matches for {type}: {len(matches)}")
        # Print all matches
        
        # Check if there are enough matches
        if len(matches) > 5:         
            # Convert keypoints to numpy arrays
            src_pts = np.float32([kp1[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Find homography using RANSAC
            # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
            M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold)
            print(f"Homography matrix for {type}:\n{M}")
            # Extract rotation angle and scale from the affine transformation matrix
            if M is not None:
                scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
                rotation_angle = np.arctan2(M[0, 1], M[0, 0]) * (180.0 / np.pi)
                print(f"Scale: {scale}, Rotation Angle: {rotation_angle} degrees")
            
            # Filter matches based on the mask
            matches_masked = [m[0] for i, m in enumerate(matches) if mask[i]]

            print(f"Number of good matches after RANSAC filtering for {type}: {len(matches_masked)}")

            # Draw the matches
            if plot_data:
                plt_ind = {'sift': 0, 'surf': 1, 'orb': 2}.get(type, 0)
                img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches_masked, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                self.filter_axes[plt_ind].imshow(img3)
                self.filter_fig.canvas.draw()
                self.filter_fig.canvas.flush_events()
        else:
            logging.warning(f"Not enough matches found for {type} in region {region}.")
            matches_masked = []
        return matches_masked
    
    # execute the matching functions
    def run(self):
        # Load SAR images
        self.load_sar_images()
        # Generate keypoints and descriptors
        self.gen_kps_and_descriptors()

        # Match keypoints and descriptors
        # 3, 9
        ind1 = 10
        ind2 = 3
        self.kp_matcher_2(ind1, ind2, type='sift', region=2)
        self.kp_matcher_2(ind1, ind2, type='orb', region=2)
        self.kp_matcher_2(ind1, ind2, type='surf', region=2)

        self.ransac_filter(ind1, ind2, type='sift', region=2, ransac_threshold=20.0)
        self.ransac_filter(ind1, ind2, type='surf', region=2, ransac_threshold=20.0)
        self.ransac_filter(ind1, ind2, type='orb', region=2, ransac_threshold=20.0)

        # self.kp_matcher_2(1, 10, type='sift', region=2)
        # self.kp_matcher_2(1, 10, type='orb', region=2)
        # self.kp_matcher_2(1, 10, type='surf', region=2)

        # self.ransac_filter(1, 10, type='sift', region=2, ransac_threshold=25.0)
        # self.ransac_filter(1, 10, type='surf', region=2, ransac_threshold=25.0)
        # self.ransac_filter(1, 10, type='orb', region=2, ransac_threshold=25.0)
        



# **********************************************************************************************
# MAIN FUNCTION
def main(args=None):
    rclpy.init(args=args)
    logging.info(f"{Colors.RED}Here we go!!!{Colors.RESET}") 
    img_match = Matcher() 
    try:
        rclpy.spin(img_match)
    except KeyboardInterrupt:
        img_match.get_logger().info(f"{Colors.RED}Keyboard Interrupt{Colors.RESET}")
        
    img_match.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except:
        logging.error('Error in the main function')
        rclpy.shutdown()
        pass
