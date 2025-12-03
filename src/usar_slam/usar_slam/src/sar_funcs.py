import numpy as np
from scipy.signal import convolve
from scipy import signal
import matplotlib.pyplot as plt
import cv2
from time import time
import math

def generate_uwb_pulse(fs, fc, frac_bw, PRF, VTX):
    """
    Generate a UWB pulse.
    
    Parameters:
    fs (float): Sampling frequency (Hz).
    fc (float): Center frequency (Hz).
    frac_bw (float): Fractional bandwidth.
    PRF (float): Pulse repetition frequency (Hz).
    VTX (float): Transmitted voltage
    
    Returns:
    ndarray: UWB pulse.
    """
    # Create time vector
    dt = 1/fs
    T = 1/PRF
    t = np.arange(-T/2, T/2, dt)
    
    i, q, e = signal.gausspulse(t, fc, frac_bw, retquad=True, retenv=True)
    # plt.plot(t, i*VTX, t, e*VTX, '--')

    return t, i*VTX

def pulse_compression(data, pulse, vis):
    """
    Find the correlation between the received signal and the pulse to determine 
    where a reflection was received.
    
    Parameters:
    data (ndarray): Matrix containing received signals, assumed to start at t = 0.
    pulse (ndarray): Transmitted pulse.
    r_min (float): Minimum range of interest (meters).
    r_max (float): Maximum range of interest (meters).
    Fs (float): Sampling frequency (Hz).
    
    Returns:
    ndarray: Correlation data within the specified range.
    """
    # pulse_length = len(pulse)
    
    # Mirror the pulse
    pulse_flip = np.flip(pulse)
    

    # Convolve each measurement with the flipped pulse
    convolved = convolve(data, pulse_flip, mode='same')
    
    # Calculate indices for the selected range
    # index_min = int(np.floor(r_min * 2 / c * Fs))
    # index_min_conv = index_min + pulse_length + 1
    # index_max = int(np.floor(r_max * 2 / c * Fs))
    # index_max_conv = index_max + pulse_length + 1
    
    # Extract relevant portion of the correlation data
    # data_correlation = convolved[:, index_min_conv:index_max_conv]
    data_correlation = convolved

    if vis:
        # Plot the received signal and the correlation result for the first measurement
        plt.figure(figsize=(12, 6))

        # Plot original received signal
        plt.subplot(2, 1, 1)
        plt.plot(data, label='Received Signal')
        plt.title('Received Signal')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend()

        # Plot correlation result
        plt.subplot(2, 1, 2)
        plt.plot(data_correlation, label='Correlation Result')
        plt.title('Correlation Result')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.tight_layout()
        plt.show()

    
    return data_correlation


def pulse_compression2(data, pulse, r_min, r_max, Fs):
    """
    Find the correlation between the received signal and the pulse to determine 
    where a reflection was received.
    
    Parameters:
    data (ndarray): Matrix containing received signals, assumed to start at t = 0.
    pulse (ndarray): Transmitted pulse.
    r_min (float): Minimum range of interest (meters).
    r_max (float): Maximum range of interest (meters).
    Fs (float): Sampling frequency (Hz).
    
    Returns:
    ndarray: Correlation data within the specified range.
    """
    # Speed of light
    c = 3e8  
    print(data.shape)
    print(data)
    # Get data dimensions
    num_measurements, data_length = data.shape
    pulse_length = len(pulse)
    
    # Mirror the pulse
    pulse_flip = np.flip(pulse)
    
    # Convolve each measurement with the flipped pulse
    print('data[0, :]:', data[0, :])
    
    convolved = np.array([convolve(data[i, :], pulse_flip, mode='full') for i in range(num_measurements)])
    
    # Calculate indices for the selected range
    index_min = int(np.floor(r_min * 2 / c * Fs))
    index_min_conv = index_min + pulse_length + 1
    index_max = int(np.floor(r_max * 2 / c * Fs))
    index_max_conv = index_max + pulse_length + 1
    
    # Extract relevant portion of the correlation data
    data_correlation = convolved[:, index_min_conv:index_max_conv]

    # Plot the received signal and the correlation result for the first measurement
    plt.figure(figsize=(12, 6))

    # Plot original received signal
    plt.subplot(2, 1, 1)
    plt.plot(data[0, :], label='Received Signal')
    plt.title('Received Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plot correlation result
    plt.subplot(2, 1, 2)
    plt.plot(data_correlation[0, :], label='Correlation Result')
    plt.title('Correlation Result')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    return data_correlation

def cart2index(cart_x, cart_y, orig_x, end_y, res): # cart_x, cart_y world coordinates
    # self.sar_end_y = self.sar_area_y - self.sar_orig_y
    x = int((cart_x + orig_x)/res)
    y = int((end_y - cart_y)/res)
    return x, y

# Normalize angle to be within -pi to pi
def normalize_angle(theta):
    return (theta + math.pi) % (2 * math.pi) - math.pi

'''
LEFT radar
Extract pixels from the radar image 
'''
def extract_pixels_radar_1(image, res, odom, r_min, r_max, half_fov, orig_x, end_y, radar_r):
    # odom_pix_y = odom[1] // res
    # odom_theta = odom[2]
    # odom_pix = np.array([odom_pix_x, odom_pix_y])

    # Calculate the range of pixels to extract
    # print('odom:', odom)
    base = [odom[0] - radar_r*np.sin(odom[2]), odom[1] + radar_r*np.cos(odom[2])] # coordinates of the radar sensor @ LEFT

    """
    def cart2index(cart_x, cart_y, orig_x, end_y, res): # cart_x, cart_y world coordinates
    # self.sar_end_y = self.sar_area_y - self.sar_orig_y
    x = int((cart_x + orig_x)/res)
    y = int((end_y - cart_y)/res)
    return x, y
    """

    '''
    # cone_left
    '''
    th_left = odom[2] + np.pi/2 + half_fov
    left_min_x = base[0] + r_min * np.cos(th_left)
    left_min_y = base[1] + r_min * np.sin(th_left)
    left_max_x = base[0] + r_max * np.cos(th_left)
    left_max_y = base[1] + r_max * np.sin(th_left)
    
    l_min_pix_x, l_min_pix_y = cart2index(left_min_x, left_min_y, orig_x, end_y, res) # cart2index(cart_x, cart_y, orig_x, end_y, res)
    l_max_pix_x, l_max_pix_y = cart2index(left_max_x, left_max_y, orig_x, end_y, res)

    '''
    # cone_right
    '''
    th_right = odom[2] + np.pi/2 - half_fov
    right_min_x = base[0] + r_min * np.cos(th_right)
    right_min_y = base[1] + r_min * np.sin(th_right)
    right_max_x = base[0] + r_max * np.cos(th_right)
    right_max_y = base[1] + r_max * np.sin(th_right)

    r_min_pix_x, r_min_pix_y = cart2index(right_min_x, right_min_y, orig_x, end_y, res) # cart2index(cart_x, cart_y, orig_x, end_y, res)
    r_max_pix_x, r_max_pix_y = cart2index(right_max_x, right_max_y, orig_x, end_y, res)

    '''
    # Extract pixels
    '''
    polygon = np.array([[l_min_pix_x, l_min_pix_y], [l_max_pix_x, l_max_pix_y], [r_max_pix_x, r_max_pix_y], [r_min_pix_x, r_min_pix_y]], dtype=np.int32)
    # print('polygon:', polygon)
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    # Get the indices of the pixels inside the triangle
    triangle_pixels = np.where(mask == 255)
    # print('triangle_pixels:', triangle_pixels)

    # Access the pixels inside the triangle (if you want to modify the image)
    # image[triangle_pixels] = [0, 255, 0]  # Fill triangle area with green

    # Display the image
    # cv2.imshow('Triangle', mask)
    # cv2.waitKey(0)  # Use waitKey(1) for continuous update
    # cv2.destroyAllWindows()

    # selected_pixels = image[mask == 255]
    # return selected_pixels
    
    return triangle_pixels, base

'''
RIGHT radar
Extract pixels from the radar image
'''
def extract_pixels_radar_2(image, res, odom, r_min, r_max, half_fov, orig_x, end_y, radar_r):
    # odom_pix_y = odom[1] // res
    # odom_theta = odom[2]
    # odom_pix = np.array([odom_pix_x, odom_pix_y])

    # Calculate the range of pixels to extract
    # print('odom:', odom)
    base = [odom[0] + radar_r*np.sin(odom[2]), odom[1] - radar_r*np.cos(odom[2])] # coordinates of the radar sensor @ RIGHT
    '''
    # cone_left
    '''
    th_left = odom[2] - np.pi/2 + half_fov
    left_min_x = base[0] + r_min * np.cos(th_left)
    left_min_y = base[1] + r_min * np.sin(th_left)
    left_max_x = base[0] + r_max * np.cos(th_left)
    left_max_y = base[1] + r_max * np.sin(th_left)
    
    l_min_pix_x, l_min_pix_y = cart2index(left_min_x, left_min_y, orig_x, end_y, res) # cart2index(cart_x, cart_y, orig_x, end_y, res)
    l_max_pix_x, l_max_pix_y = cart2index(left_max_x, left_max_y, orig_x, end_y, res)

    '''
    # cone_right
    '''
    th_right = odom[2] - np.pi/2 - half_fov
    right_min_x = base[0] + r_min * np.cos(th_right)
    right_min_y = base[1] + r_min * np.sin(th_right)
    right_max_x = base[0] + r_max * np.cos(th_right)
    right_max_y = base[1] + r_max * np.sin(th_right)

    r_min_pix_x, r_min_pix_y = cart2index(right_min_x, right_min_y, orig_x, end_y, res) # cart2index(cart_x, cart_y, orig_x, end_y, res)
    r_max_pix_x, r_max_pix_y = cart2index(right_max_x, right_max_y, orig_x, end_y, res)

    '''
    # Extract pixels
    '''
    polygon = np.array([[l_min_pix_x, l_min_pix_y], [l_max_pix_x, l_max_pix_y], [r_max_pix_x, r_max_pix_y], [r_min_pix_x, r_min_pix_y]], dtype=np.int32)
    # print('polygon:', polygon)
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    # Get the indices of the pixels inside the triangle
    triangle_pixels = np.where(mask == 255)
    # print('triangle_pixels:', triangle_pixels)

    # Access the pixels inside the triangle (if you want to modify the image)
    # image[triangle_pixels] = [0, 255, 0]  # Fill triangle area with green

    # Display the image
    # cv2.imshow('Triangle', mask)
    # cv2.waitKey(0)  # Use waitKey(1) for continuous update
    # cv2.destroyAllWindows()

    # selected_pixels = image[mask == 255]
    # return selected_pixels
    return triangle_pixels, base

def extract_img_region(image, res, odom0, odom1, r_max, orig_x, end_y):
    # odom_pix_y = odom[1] // res
    # odom_theta = odom[2]
    # odom_pix = np.array([odom_pix_x, odom_pix_y])
    odom0[2] = normalize_angle(odom0[2])
    odom1[2] = normalize_angle(odom1[2])
    # Calculate the range of pixels to extract
    '''
    # odom0
    '''
    odom0_left_x = odom0[0] + r_max * np.cos(odom0[2] + np.pi/2)
    odom0_left_y = odom0[1] + r_max * np.sin(odom0[2] + np.pi/2)
    odom0_right_x = odom0[0] + r_max * np.cos(odom0[2] - np.pi/2)
    odom0_right_y = odom0[1] + r_max * np.sin(odom0[2] - np.pi/2)

    odom0_left_pix_x, odom0_left_pix_y = cart2index(odom0_left_x, odom0_left_y, orig_x, end_y, res) # cart2index(cart_x, cart_y, orig_x, end_y, res)
    odom0_right_pix_x, odom0_right_pix_y = cart2index(odom0_right_x, odom0_right_y, orig_x, end_y, res)

    '''
    # odom1
    '''
    odom1_left_x = odom1[0] + r_max * np.cos(odom1[2] + np.pi/2)
    odom1_left_y = odom1[1] + r_max * np.sin(odom1[2] + np.pi/2)
    odom1_right_x = odom1[0] + r_max * np.cos(odom1[2] - np.pi/2)
    odom1_right_y = odom1[1] + r_max * np.sin(odom1[2] - np.pi/2)
    odom1_left_pix_x, odom1_left_pix_y = cart2index(odom1_left_x, odom1_left_y, orig_x, end_y, res) # cart2index(cart_x, cart_y, orig_x, end_y, res)
    odom1_right_pix_x, odom1_right_pix_y = cart2index(odom1_right_x, odom1_right_y, orig_x, end_y, res)

    '''
    # Extract pixels
    '''
    min_pix_x = min(odom0_left_pix_x, odom1_left_pix_x, odom0_right_pix_x, odom1_right_pix_x)
    min_pix_y = min(odom0_left_pix_y, odom1_left_pix_y, odom0_right_pix_y, odom1_right_pix_y)
    max_pix_x = max(odom0_left_pix_x, odom1_left_pix_x, odom0_right_pix_x, odom1_right_pix_x)
    max_pix_y = max(odom0_left_pix_y, odom1_left_pix_y, odom0_right_pix_y, odom1_right_pix_y)
    
    '''''
    polygon = np.array([[max_pix_x, max_pix_y], [min_pix_x, max_pix_y], [min_pix_x, min_pix_y], [max_pix_x, min_pix_y]], dtype=np.int32)
    # print('polygon:', polygon)
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    # Get the indices of the pixels inside the triangle
    region_pixels = np.where(mask == 255)
    # print('region_pixels:', region_pixels)
    # Access the pixels inside the triangle (if you want to modify the image)
    image[region_pixels] = [0, 255, 0]  # Fill triangle area with green
    '''

    # Find the odom0 transformation matrix w.r.t. the image origin
    '''
    img_orig_x = min(odom0_left_x, odom1_left_x, odom0_right_x, odom1_right_x)
    img_orig_y = min(odom0_left_y, odom1_left_y, odom0_right_y, odom1_right_y)
    odom0_dx = odom0[0] - img_orig_x
    odom0_dy = odom0[1] - img_orig_y
    odom0_theta = odom0[2]
    T_odom0 = np.array([[np.cos(odom0_theta), -np.sin(odom0_theta), odom0_dx],
                        [np.sin(odom0_theta), np.cos(odom0_theta), odom0_dy],
                        [0, 0, 1]])
    '''
    img_orig_x = min(odom0_left_x, odom1_left_x, odom0_right_x, odom1_right_x)
    img_orig_y = max(odom0_left_y, odom1_left_y, odom0_right_y, odom1_right_y)
    odom0_dx = odom0[0] - img_orig_x
    odom0_dy = img_orig_y - odom0[1]  # Invert y-axis for image coordinates
    odom0_theta = -odom0[2]
    T_odom0 = np.array([[np.cos(odom0_theta), -np.sin(odom0_theta), odom0_dx],
                        [np.sin(odom0_theta), np.cos(odom0_theta), odom0_dy],
                        [0, 0, 1]])
    '''
    # Display the image
    cv2.imshow('Triangle', mask)
    cv2.waitKey(0)  # Use waitKey(1) for continuous update
    # cv2.destroyAllWindows()
    '''
    # print('min_max:', min_pix_x, max_pix_x, min_pix_y, max_pix_y)
    min_pix_x = max(0, min_pix_x)
    min_pix_y = max(0, min_pix_y)

    return min_pix_x, max_pix_x, min_pix_y, max_pix_y, T_odom0

""" For two images in two regions """
def SIFT_extract_keypoints2(cropped_sar_1, cropped_sar_2):
        # Initialize SIFT detector
        sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.01, edgeThreshold=10, sigma=1.6)

        # Detect keypoints and compute descriptors for cropped_sar_1
        keypoints_1, descriptors_1 = sift.detectAndCompute(cropped_sar_1, None)

        # Detect keypoints and compute descriptors for cropped_sar_2
        keypoints_2, descriptors_2 = sift.detectAndCompute(cropped_sar_2, None)

        return keypoints_1, descriptors_1, keypoints_2, descriptors_2   
def ORB_extract_keypoints2(cropped_sar_1, cropped_sar_2):
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=15, WTA_K=4)

    # Detect keypoints and compute descriptors for cropped_sar_1
    keypoints_1, descriptors_1 = orb.detectAndCompute(cropped_sar_1, None)

    # Detect keypoints and compute descriptors for cropped_sar_2
    keypoints_2, descriptors_2 = orb.detectAndCompute(cropped_sar_2, None)

    return keypoints_1, descriptors_1, keypoints_2, descriptors_2
def SURF_extract_keypoints2(cropped_sar_1, cropped_sar_2):
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
def AKAZE_extract_keypoints2(cropped_sar_1, cropped_sar_2):
    # Initialize AKAZE detector
    akaze = cv2.AKAZE_create(threshold= 0.0005, max_points=200, descriptor_type=cv2.AKAZE_DESCRIPTOR_KAZE) # cv2.AKAZE_DESCRIPTOR_MLDB [Change matching distance accordingly]
    # Detect keypoints and compute descriptors for cropped_sar_1
    keypoints_1, descriptors_1 = akaze.detectAndCompute(cropped_sar_1, None)
    # Detect keypoints and compute descriptors for cropped_sar_2
    keypoints_2, descriptors_2 = akaze.detectAndCompute(cropped_sar_2, None)

    return keypoints_1, descriptors_1, keypoints_2, descriptors_2

"""
ONLY For one image in one region
"""
def SIFT_extract_keypoints(cropped_sar):
        # Initialize SIFT detector
        sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.01, edgeThreshold=10, sigma=1.6)

        # Detect keypoints and compute descriptors for cropped_sar_1
        keypoints_1, descriptors_1 = sift.detectAndCompute(cropped_sar, None)

        return keypoints_1, descriptors_1
def ORB_extract_keypoints(cropped_sar):
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=15, WTA_K=4)

    # Detect keypoints and compute descriptors for cropped_sar_1
    keypoints_1, descriptors_1 = orb.detectAndCompute(cropped_sar, None)

    return keypoints_1, descriptors_1
def SURF_extract_keypoints(cropped_sar):        
    # Initialize SURF detector
    try:
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400, nOctaves=4, nOctaveLayers=2, extended=False, upright=False)
    except AttributeError:
        raise ImportError("SURF is not available. Please install opencv-contrib-python.")

    # Detect keypoints and compute descriptors for cropped_sar_1
    keypoints_1, descriptors_1 = surf.detectAndCompute(cropped_sar, None)

    return keypoints_1, descriptors_1

def AKAZE_extract_keypoints(cropped_sar):
    # Initialize AKAZE detector
    akaze = cv2.AKAZE_create(threshold= 0.0005, max_points=200, descriptor_type=cv2.AKAZE_DESCRIPTOR_KAZE) # cv2.AKAZE_DESCRIPTOR_MLDB [Change matching distance accordingly]
    # Detect keypoints and compute descriptors for cropped_sar_1
    keypoints_1, descriptors_1 = akaze.detectAndCompute(cropped_sar, None)

    return keypoints_1, descriptors_1

def find_homography(kp1, kp2, img1, img2, matches, type, ransac_thresh, match_fig, match_ax, plot_data):
    matches_masked = []
    M = None
    scale = -1
    rotation_angle = 0
    displacement = 0
    if len(matches) > 5:         
        # Convert keypoints to numpy arrays
        src_pts = np.float32([kp1[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography using RANSAC
        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        M[1, 2] = M[1, 2] # Invert y-axis for image coordinates
        print(f"Homography matrix (y-inverted, orig top-left) for {type}:\n{M}")
        # Extract rotation angle and scale from the affine transformation matrix
        if M is not None:
            scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
            rotation_angle = np.arctan2(M[1, 0], M[0, 0])
            displacement = np.array([M[0, 2], M[1, 2]]) # (-) because the image y-axis is inverted
            print(f"Scale: {scale}, Rotation Angle: {rotation_angle} radians = {np.degrees(rotation_angle)} degrees")
        
        # Filter matches based on the mask
        matches_masked = [m[0] for i, m in enumerate(matches) if mask[i]]

        print(f"Number of good matches after RANSAC filtering for {type}: {len(matches_masked)}")

        # Draw the matches
        if plot_data:
            img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches_masked, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            match_ax.imshow(img3)
            match_fig.canvas.draw()
            match_fig.canvas.flush_events()
    return matches_masked, len(matches_masked), M, scale, rotation_angle, displacement

def match_kp(kp1, kp2, des1, des2, img1, img2, match_fig, match_ax, plot_data):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good_matches.append([m])

    if plot_data:
        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        match_ax.imshow(img3)
        match_fig.canvas.draw()
        match_fig.canvas.flush_events()
    return good_matches

'''
LEFT SAR
Extract pixels using OpenCV methods
'''
def add_sar_radar_1(odom, obs_radar, img_dist_x, img_dist_y, img, res, data_left, half_fov, sar_orig_x, sar_end_y, radar_r, samp_size):
    # start_time = time.time()

    # Extract pixel coordinates & sensor base
    pixels, sensor_base = extract_pixels_radar_1(img, res, odom, 400, 3000, half_fov, sar_orig_x, sar_end_y, radar_r)
    # print('Time taken to extract pixels:', time.time() - start_time)

    # Convert pixel indices to real-world coordinates (vectorized)
    p_y, p_x = pixels
    pix_x = img_dist_x[0, p_x]  # Vectorized lookup
    pix_y = img_dist_y[p_y, 0]  # Vectorized lookup

    # Compute distance squared (vectorized)
    d_sqr = np.hypot(pix_x - sensor_base[0], pix_y - sensor_base[1])

    # Vectorized lookup for nearest values
    val = np.interp(d_sqr, np.arange(len(obs_radar)) * samp_size, obs_radar)

    # Vectorized image update
    img[p_y, p_x] += val

    data_left["px"].append(p_x)
    data_left["py"].append(p_y)
    data_left["val"].append(val)
    data_left["odom"].append(odom)

    # print('Time taken to add SAR:', time.time() - start_time)
    return img

def add_sar_radar_1_regen(odom, obs_radar, img_dist_x, img_dist_y, img, mask_img, res, half_fov, sar_orig_x, sar_end_y, radar_r, samp_size):
    # start_time = time.time()

    # Extract pixel coordinates & sensor base
    pixels, sensor_base = extract_pixels_radar_1(img, res, odom, 400, 3000, half_fov, sar_orig_x, sar_end_y, radar_r)
    # print('Time taken to extract pixels:', time.time() - start_time)

    # Convert pixel indices to real-world coordinates (vectorized)
    p_y, p_x = pixels
    # pix_x = img_dist_x[0, p_x]  # Vectorized lookup
    # pix_y = img_dist_y[p_y, 0]  # Vectorized lookup

    # Vectorized mask filtering for efficiency
    mask = img[p_y, p_x] <= 600
    pix_x = img_dist_x[0, p_x[mask]]
    pix_y = img_dist_y[p_y[mask], 0]
    p_x = p_x[mask]
    p_y = p_y[mask]

    """
    pix_x = np.array([])
    pix_y = np.array([])

    for i in range(len(p_x)):
        if img[p_y[i], p_x[i]] <= 5:
            pix_x = np.append(pix_x, img_dist_x[0, p_x[i]]) # Vectorized lookup
            pix_y = np.append(pix_y, img_dist_y[p_y[i], 0]) # Vectorized lookup
    """

    # Apply mask to filter pixels
    mask_img[p_y, p_x] += 1  # Update image to indicate pixels have been processed

    # Compute distance squared (vectorized)
    d_sqr = np.hypot(pix_x - sensor_base[0], pix_y - sensor_base[1])

    # Vectorized lookup for nearest values
    val = np.interp(d_sqr, np.arange(len(obs_radar)) * samp_size, obs_radar)

    # Vectorized image update
    img[p_y, p_x] += val

    # print('Time taken to add SAR:', time.time() - start_time)
    return img, mask_img

'''
RIGHT SAR
Extract pixels using OpenCV methods
'''
def add_sar_radar_2(odom, obs_radar, img_dist_x, img_dist_y, img, res, data_right, half_fov, sar_orig_x, sar_end_y, radar_r, samp_size):
    # start_time = time.time()

    # Extract pixel coordinates & sensor base
    pixels, sensor_base = extract_pixels_radar_2(img, res, odom, 400, 3000, half_fov, sar_orig_x, sar_end_y, radar_r)
    # print('Time taken to extract pixels:', time.time() - start_time)

    # Convert pixel indices to real-world coordinates (vectorized)
    p_y, p_x = pixels
    pix_x = img_dist_x[0, p_x]  # Vectorized lookup
    pix_y = img_dist_y[p_y, 0]  # Vectorized lookup

    # Compute distance squared (vectorized)
    d_sqr = np.hypot(pix_x - sensor_base[0], pix_y - sensor_base[1])

    # Vectorized lookup for nearest values using numpy interpolation
    val = np.interp(d_sqr, np.arange(len(obs_radar)) * samp_size, obs_radar)

    # Vectorized image update
    img[p_y, p_x] += val

    data_right["px"].append(p_x)
    data_right["py"].append(p_y)
    data_right["val"].append(val)
    data_right["odom"].append(odom)

    # print('Time taken to add SAR:', time.time() - start_time)
    return img

def add_sar_radar_2_regen(odom, obs_radar, img_dist_x, img_dist_y, img, mask_img, res, half_fov, sar_orig_x, sar_end_y, radar_r, samp_size):
    # start_time = time.time()

    # Extract pixel coordinates & sensor base
    pixels, sensor_base = extract_pixels_radar_2(img, res, odom, 400, 3000, half_fov, sar_orig_x, sar_end_y, radar_r)
    # print('Time taken to extract pixels:', time.time() - start_time)

    # Convert pixel indices to real-world coordinates (vectorized)
    p_y, p_x = pixels
    # pix_x = img_dist_x[0, p_x]  # Vectorized lookup
    # pix_y = img_dist_y[p_y, 0]  # Vectorized lookup

    # Vectorized mask filtering for efficiency
    mask = mask_img[p_y, p_x] <= 600
    pix_x = img_dist_x[0, p_x[mask]]
    pix_y = img_dist_y[p_y[mask], 0]
    p_x = p_x[mask]
    p_y = p_y[mask]

    """
    pix_x = np.array([])
    pix_y = np.array([])

    for i in range(len(p_x)):
        if mask_img[p_y[i], p_x[i]] <= 5:
            pix_x = np.append(pix_x, img_dist_x[0, p_x[i]]) # Vectorized lookup
            pix_y = np.append(pix_y, img_dist_y[p_y[i], 0]) # Vectorized lookup
    """

    # Apply mask to filter pixels
    mask_img[p_y, p_x] += 1  # Update mask image to indicate pixels have been processed

    # Compute distance squared (vectorized)
    d_sqr = np.hypot(pix_x - sensor_base[0], pix_y - sensor_base[1])

    # Vectorized lookup for nearest values using numpy interpolation
    val = np.interp(d_sqr, np.arange(len(obs_radar)) * samp_size, obs_radar)

    # Vectorized image update
    img[p_y, p_x] += val

    # print('Time taken to add SAR:', time.time() - start_time)
    return img, mask_img

def SIFT_extract_and_display_keypoints(cropped_sar_1, cropped_sar_2, sar_ax1, sar_ax2, img_fig):
    # Initialize SIFT detector
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.01, edgeThreshold=10, sigma=1.6)

    # Normalize and convert cropped_sar_1 and cropped_sar_2 to uint8
    cropped_sar_1 = cv2.normalize(cropped_sar_1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cropped_sar_2 = cv2.normalize(cropped_sar_2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Detect keypoints and compute descriptors for cropped_sar_1
    keypoints_1, descriptors_1 = sift.detectAndCompute(cropped_sar_1, None)

    # Draw keypoints on the image for visualization
    cropped_sar_1_with_keypoints = cv2.drawKeypoints(
        cropped_sar_1, keypoints_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Display the image with keypoints
    sar_ax1.imshow(cropped_sar_1_with_keypoints, cmap='jet', animated=True)

    # Detect keypoints and compute descriptors for cropped_sar_2
    keypoints_2, descriptors_2 = sift.detectAndCompute(cropped_sar_2, None)

    # Draw keypoints on the image for visualization
    cropped_sar_2_with_keypoints = cv2.drawKeypoints(
        cropped_sar_2, keypoints_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Display the image with keypoints
    sar_ax2.imshow(cropped_sar_2_with_keypoints, cmap='jet', animated=True)

    img_fig.canvas.draw()
    img_fig.canvas.flush_events()

    return 0   

def ORB_extract_and_display_keypoints(cropped_sar_1, cropped_sar_2, sar_ax1, sar_ax2, img_fig):
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=5000)

    # Normalize and convert cropped_sar_1 and cropped_sar_2 to uint8
    cropped_sar_1 = cv2.normalize(cropped_sar_1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cropped_sar_2 = cv2.normalize(cropped_sar_2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Detect keypoints and compute descriptors for cropped_sar_1
    keypoints_1, descriptors_1 = orb.detectAndCompute(cropped_sar_1, None)

    # Draw keypoints on the image for visualization
    cropped_sar_1_with_keypoints = cv2.drawKeypoints(
        cropped_sar_1, keypoints_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Display the image with keypoints
    sar_ax1.imshow(cropped_sar_1_with_keypoints, cmap='jet', animated=True)

    # Detect keypoints and compute descriptors for cropped_sar_2
    keypoints_2, descriptors_2 = orb.detectAndCompute(cropped_sar_2, None)

    # Draw keypoints on the image for visualization
    cropped_sar_2_with_keypoints = cv2.drawKeypoints(
        cropped_sar_2, keypoints_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Display the image with keypoints
    sar_ax2.imshow(cropped_sar_2_with_keypoints, cmap='jet', animated=True)  

    img_fig.canvas.draw()
    img_fig.canvas.flush_events()

    return 0

def SURF_extract_and_display_keypoints(cropped_sar_1, cropped_sar_2, sar_ax1, sar_ax2, img_fig):
    # Initialize SURF detector
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400, nOctaves=4, nOctaveLayers=2, extended=False, upright=False)

    # Normalize and convert cropped_sar_1 and cropped_sar_2 to uint8
    cropped_sar_1 = cv2.normalize(cropped_sar_1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cropped_sar_2 = cv2.normalize(cropped_sar_2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Detect keypoints and compute descriptors for cropped_sar_1
    keypoints_1, descriptors_1 = surf.detectAndCompute(cropped_sar_1, None)

    # Draw keypoints on the image for visualization
    cropped_sar_1_with_keypoints = cv2.drawKeypoints(
        cropped_sar_1, keypoints_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Display the image with keypoints
    sar_ax1.imshow(cropped_sar_1_with_keypoints, cmap='jet', animated=True)

    # Detect keypoints and compute descriptors for cropped_sar_2
    keypoints_2, descriptors_2 = surf.detectAndCompute(cropped_sar_2, None)

    # Draw keypoints on the image for visualization
    cropped_sar_2_with_keypoints = cv2.drawKeypoints(
        cropped_sar_2, keypoints_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Display the image with keypoints
    sar_ax2.imshow(cropped_sar_2_with_keypoints, cmap='jet', animated=True)  

    img_fig.canvas.draw()
    img_fig.canvas.flush_events()

    return 0

def AKAZE_extract_and_display_keypoints(cropped_sar_1, cropped_sar_2, ax1, ax2, img_fig):
    start_time = time.time()
    # Initialize AKAZE detector
    akaze = cv2.AKAZE_create(threshold= 0.0005, max_points=200, descriptor_type=cv2.AKAZE_DESCRIPTOR_KAZE) # cv2.AKAZE_DESCRIPTOR_MLDB [Change matching distance accordingly]
    # Detect keypoints and compute descriptors for cropped_sar_1
    keypoints_1, descriptors_1 = akaze.detectAndCompute(cropped_sar_1, None)
    # Detect keypoints and compute descriptors for cropped_sar_2
    keypoints_2, descriptors_2 = akaze.detectAndCompute(cropped_sar_2, None)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time for AKAZE keypoint detection and descriptor computation: {elapsed_time:.8f} seconds")

    # Draw keypoints on the images
    img1 = cv2.drawKeypoints(cropped_sar_1, keypoints_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2 = cv2.drawKeypoints(cropped_sar_2, keypoints_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the images with keypoints
    ax1.imshow(img1)
    ax2.imshow(img2)
    img_fig.canvas.draw()
    img_fig.canvas.flush_events()

    return 0