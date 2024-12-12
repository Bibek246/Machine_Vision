#Bibek Kumar Sharma
#CS455
import cv2
import numpy as np

def preprocess_frame(frame):
    """
    Preprocess the frame by converting to grayscale and applying Gaussian blur.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    return blurred_frame

def detect_motion(preprocessed_frame, previous_frame):
    """
    Detect motion by comparing the current preprocessed frame with the previous one.
    """
    # Compute the absolute difference between the current frame and previous frame
    frame_delta = cv2.absdiff(previous_frame, preprocessed_frame)

    # Threshold the delta image to get a binary image
    _, threshold_frame = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)

    # Dilate the threshold image to fill in holes, making it easier to detect motion
    dilated_frame = cv2.dilate(threshold_frame, None, iterations=2)

    return dilated_frame

def visualize_motion(frame, motion_mask):
    """
    Visualize the detected motion by finding the bounding box based on non-zero pixels in the motion mask.
    """
    # Get the coordinates of non-zero pixels (motion)
    non_zero_points = np.column_stack(np.where(motion_mask > 0))

    if len(non_zero_points) > 0:
        # Calculate a bounding box around the non-zero pixels (motion areas)
        x, y, w, h = cv2.boundingRect(non_zero_points)
        
        # Draw a rectangle around the detected motion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return frame

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path

    # Initialize video writer to record the output in MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
    out = cv2.VideoWriter('motion_detection_output.mp4', fourcc, 20.0, (640, 480))

    # Read the first frame to initialize the previous frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error reading video stream")
        return

    previous_frame = preprocess_frame(first_frame)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the current frame
        preprocessed_frame = preprocess_frame(frame)

        # Detect motion between the previous and current frames
        motion_mask = detect_motion(preprocessed_frame, previous_frame)

        # Visualize motion by drawing bounding boxes around the motion areas
        motion_frame = visualize_motion(frame, motion_mask)

        # Show the frame with motion visualization
        cv2.imshow('Motion Detector', motion_frame)

        # Write the frame to the output video
        out.write(motion_frame)

        # Update the previous frame
        previous_frame = preprocessed_frame

        # Press 'q' to exit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and file resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
