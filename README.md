# Feature Detection and Matching for Object Detection

## Overview
This project implements feature detection and matching techniques to compare images using three different feature extraction methods: **SIFT, ORB, and BRISK**. It applies two types of feature matching methods: **Brute-Force (BF) Matcher** and **FLANN-based Matcher** (for SIFT).

The program takes in a **master image** and compares it with two other images using these techniques, displaying and saving the matched keypoints.

## Features
- **Feature Detectors:**
  - **SIFT (Scale-Invariant Feature Transform)**
  - **ORB (Oriented FAST and Rotated BRIEF)**
  - **BRISK (Binary Robust Invariant Scalable Keypoints)**
  
- **Feature Matching Methods:**
  - **Brute-Force Matcher (BF)**
  - **FLANN-Based Matcher (for SIFT)**

- **Visualization:**
  - Draws and saves **top 20** feature matches for each comparison.

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install opencv-python matplotlib numpy
Usage
Prepare the images

Place three images in the same directory as the script:
image1.jpg (Master image)
image2.jpg (Comparison image 1)
image3.jpg (Comparison image 2)

Run the script

**python features.py**

**## How It Works**
Load images and check their validity.
Detect keypoints and descriptors using SIFT, ORB, and BRISK.
Match features using:
**Brute-Force Matcher (BF) for all detectors.
FLANN Matcher (only for SIFT).
**
Draw and save matches to visualize results.
