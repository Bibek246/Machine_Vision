#Bibek Kumar Sharma

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
master_image = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('image3.jpg', cv2.IMREAD_GRAYSCALE)

# Check if images loaded
assert master_image is not None, "Master image not found"
assert image2 is not None, "Second image not found"
assert image3 is not None, "Third image not found"

# SIFT Feature Detector
def detect_sift(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# ORB Feature Detector
def detect_orb(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

# BRISK Feature Detector (replacing AKAZE)
def detect_brisk(image):
    brisk = cv2.BRISK_create()
    keypoints, descriptors = brisk.detectAndCompute(image, None)
    return keypoints, descriptors

# Brute Force Matcher
def brute_force_matcher(desc1, desc2, norm_type=cv2.NORM_L2):
    bf = cv2.BFMatcher(norm_type, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

# FLANN Matcher (only for SIFT because it uses floating-point descriptors)
def flann_matcher(desc1, desc2):
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE for SIFT
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches

# Drawing Matches
def draw_matches(img1, kp1, img2, kp2, matches, method_name, pair_name):
    result_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(10, 5))
    plt.title(f'{method_name} - Matches')
    plt.imshow(result_img)
    plt.axis('off')
    plt.savefig(f'{method_name}_{pair_name}_matches.png')
    plt.show()

# Running Feature Detection and Matching
def run_feature_detection_and_matching(master_img, img2, img3):
    # SIFT
    sift_kp1, sift_desc1 = detect_sift(master_img)
    sift_kp2, sift_desc2 = detect_sift(img2)
    sift_kp3, sift_desc3 = detect_sift(img3)
    
    # ORB
    orb_kp1, orb_desc1 = detect_orb(master_img)
    orb_kp2, orb_desc2 = detect_orb(img2)
    orb_kp3, orb_desc3 = detect_orb(img3)
    
    # BRISK
    brisk_kp1, brisk_desc1 = detect_brisk(master_img)e
    brisk_kp2, brisk_desc2 = detect_brisk(img2)
    brisk_kp3, brisk_desc3 = detect_brisk(img3)

    # Matching SIFT Features (Use FLANN for SIFT as it has floating-point descriptors)
    sift_bf_matches_2 = brute_force_matcher(sift_desc1, sift_desc2, cv2.NORM_L2)
    sift_bf_matches_3 = brute_force_matcher(sift_desc1, sift_desc3, cv2.NORM_L2)
    draw_matches(master_img, sift_kp1, img2, sift_kp2, sift_bf_matches_2, "SIFT-BF", "Image2")
    draw_matches(master_img, sift_kp1, img3, sift_kp3, sift_bf_matches_3, "SIFT-BF", "Image3")
    
    sift_flann_matches_2 = flann_matcher(sift_desc1, sift_desc2)
    sift_flann_matches_3 = flann_matcher(sift_desc1, sift_desc3)
    draw_matches(master_img, sift_kp1, img2, sift_kp2, sift_flann_matches_2, "SIFT-FLANN", "Image2")
    draw_matches(master_img, sift_kp1, img3, sift_kp3, sift_flann_matches_3, "SIFT-FLANN", "Image3")
    
    # Matching ORB Features (Use Brute-Force for binary descriptors)
    orb_bf_matches_2 = brute_force_matcher(orb_desc1, orb_desc2, cv2.NORM_HAMMING)
    orb_bf_matches_3 = brute_force_matcher(orb_desc1, orb_desc3, cv2.NORM_HAMMING)
    draw_matches(master_img, orb_kp1, img2, orb_kp2, orb_bf_matches_2, "ORB-BF", "Image2")
    draw_matches(master_img, orb_kp1, img3, orb_kp3, orb_bf_matches_3, "ORB-BF", "Image3")
    
    orb_bf2_matches_2 = brute_force_matcher(orb_desc1, orb_desc2, cv2.NORM_HAMMING)
    orb_bf2_matches_3 = brute_force_matcher(orb_desc1, orb_desc3, cv2.NORM_HAMMING)
    draw_matches(master_img, orb_kp1, img2, orb_kp2, orb_bf2_matches_2, "ORB-BF2", "Image2")
    draw_matches(master_img, orb_kp1, img3, orb_kp3, orb_bf2_matches_3, "ORB-BF2", "Image3")
    
    # Matching BRISK Features (Use Brute-Force for binary descriptors)
    brisk_bf_matches_2 = brute_force_matcher(brisk_desc1, brisk_desc2, cv2.NORM_HAMMING)
    brisk_bf_matches_3 = brute_force_matcher(brisk_desc1, brisk_desc3, cv2.NORM_HAMMING)
    draw_matches(master_img, brisk_kp1, img2, brisk_kp2, brisk_bf_matches_2, "BRISK-BF", "Image2")
    draw_matches(master_img, brisk_kp1, img3, brisk_kp3, brisk_bf_matches_3, "BRISK-BF", "Image3")

    brisk_bf2_matches_2 = brute_force_matcher(brisk_desc1, brisk_desc2, cv2.NORM_HAMMING)
    brisk_bf2_matches_3 = brute_force_matcher(brisk_desc1, brisk_desc3, cv2.NORM_HAMMING)
    draw_matches(master_img, brisk_kp1, img2, brisk_kp2, brisk_bf2_matches_2, "BRISK-BF2", "Image2")
    draw_matches(master_img, brisk_kp1, img3, brisk_kp3, brisk_bf2_matches_3, "BRISK-BF2", "Image3")

# Execute the full matching process
run_feature_detection_and_matching(master_image, image2, image3)
