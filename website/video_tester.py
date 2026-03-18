import cv2
import io
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained deep learning-based face detector
prototxt_path = '/Users/kunal/College/Capstone/website/deploy.prototxt.txt'
caffemodel_path = '/Users/kunal/College/Capstone/website/res10_300x300_ssd_iter_140000.caffemodel'

net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

def video_processor(video_path, num_frames=300):
    # Create a VideoCapture object from the file path
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get the frames per second (FPS) of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Frames per second: {fps}")

    frames = []

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to your desired size
        # frame = cv2.resize(frame, (72, 72))

        # Get frame dimensions
        # (h, w) = frame.shape[:2]
        (h,w)=(300,300)
        # Prepare the frame for face detection
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)))
        
        # Pass the frame through the deep learning-based detector
        net.setInput(blob)
        detections = net.forward()

        # Extract faces from the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.8:  # Adjust confidence threshold as needed
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]
                frames.append(face)

    cap.release()
    frames_array = np.array(frames)
    resized_frames=[]
    for frames in frames_array:
        frames=cv2.resize(frames, (72, 72))
        resized_frames.append(frames)
        
    resized_frames = np.array(resized_frames)

    return resized_frames


r=video_processor('/Users/kunal/College/Capstone/website/temp_video.mp4')
print(r.shape)
import matplotlib.pyplot as plt

# Assuming you have a NumPy array 'resized_frames' with shape (300, 72, 72, 3)
frame_to_display = r[10]  # Change the index to the frame you want to display

# Display the frame
plt.imshow(frame_to_display)
plt.axis('off')  # Optional: Turn off axis labels
plt.show()