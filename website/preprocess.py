import cv2
import io
import numpy as np
import matplotlib.pyplot as plt
def video_processor(video_path, num_frames=300):
    # Create a VideoCapture object from the file path
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        # Get the frames per second (FPS) of the video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Frames per second: {fps}")

    frames = []

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Append the frame to the list
        frame=cv2.resize(frame,(72,72))
        frames.append(frame)
    
    cap.release()
    frames_array = np.array(frames)
    

    return frames_array
