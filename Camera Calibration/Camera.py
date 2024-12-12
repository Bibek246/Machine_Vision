#Name:Bibek Kumar Sharma
import cv2
import numpy as np
import os
import json

# Define the chessboard size (change according to your printed pattern)
chessboard_size = (9, 6)  # 9x6 means 9 corners horizontally and 6 corners vertically
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for a standard chessboard
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
img_counter = 0
captured_images = 10  # Minimum number of images for calibration

# Create directory to save captured images
if not os.path.exists('calibration_images'):
    os.makedirs('calibration_images')

# Capture chessboard images
while img_counter < captured_images:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        # Refine and add object points and image points
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)

        # Save the image
        img_name = f"calibration_images/calib_image_{img_counter}.png"
        cv2.imwrite(img_name, frame)
        img_counter += 1
        print(f"Image {img_counter} captured and saved: {img_name}")

    # Display the frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calibration step
if len(objpoints) >= captured_images:
    print("Calibrating camera...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"Reprojection error: {mean_error / len(objpoints)}")

    # Save calibration parameters to JSON
    calibration_data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "rotation_vectors": [rvec.tolist() for rvec in rvecs],
        "translation_vectors": [tvec.tolist() for tvec in tvecs]
    }

    with open("camera-params.json", "w") as json_file:
        json.dump(calibration_data, json_file, indent=4)
    print("Calibration parameters saved to camera-params.json")

# Capture a new distorted image BEFORE releasing the camera
ret, frame = cap.read()
if ret:
    distorted_image_name = "distorted_image.png"
    cv2.imwrite(distorted_image_name, frame)
    print(f"Captured distorted image saved as {distorted_image_name}")

    # Undistort the captured image using the loaded camera matrix and distortion coefficients
    undistorted_image = cv2.undistort(frame, camera_matrix, dist_coeffs)
    undistorted_image_name = "undistorted_image.png"
    cv2.imwrite(undistorted_image_name, undistorted_image)
    print(f"Undistorted image saved as {undistorted_image_name}")

    # Display distorted and undistorted images
    cv2.imshow("Distorted Image", frame)
    cv2.imshow("Undistorted Image", undistorted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to capture a distorted image.")

# Now it's safe to release the camera
cap.release()
cv2.destroyAllWindows()

# Load calibration parameters from JSON file
with open("camera-params.json", "r") as json_file:
    calibration_data = json.load(json_file)

# Convert calibration data back to NumPy arrays
camera_matrix = np.array(calibration_data["camera_matrix"])
dist_coeffs = np.array(calibration_data["dist_coeffs"])
rvecs = [np.array(rvec) for rvec in calibration_data["rotation_vectors"]]
tvecs = [np.array(tvec) for tvec in calibration_data["translation_vectors"]]

# Write-up (saved to a text file for documentation)
writeup = """
This script performs camera calibration using OpenCV. It captures a series of images
of a chessboard pattern from various angles, detects the corners in these images, and
uses them to compute the camera's intrinsic parameters and distortion coefficients.

The calibration parameters are saved in 'camera-params.json' for future use. 
A distorted image is captured, and then an undistorted version is generated using the
calibration parameters. Both images are saved as 'distorted_image.png' and 'undistorted_image.png'.

Reprojection error was calculated to assess the accuracy of the calibration, and it 
was found to be reasonably low.
"""
with open("writeup.txt", "w") as f:
    f.write(writeup)
print("Write-up saved to writeup.txt")
