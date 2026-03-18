from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os

# Define the maximum video length in seconds
max_video_length = 10

# Replace 'your_directory' with the path to your 'celeb_df' directory
base_directory = '/Users/kunal/Downloads/Celeb_DF(V2) dataset'

# Function to process videos in a directory
def process_videos_in_directory(directory):
    for video_name in os.listdir(directory):
        video_path = os.path.join(directory, video_name)
        
        # Check if it's a file and not a subdirectory
        if os.path.isfile(video_path):
            video = video_name.split('.')[0]  # Remove the file extension
            output_path = os.path.join(directory, f"{video}_trimmed.mp4")
            
            # Trim the video to the specified length
            ffmpeg_extract_subclip(video_path, 0, max_video_length, targetname=output_path)

# Process videos in the 'real' and 'fake' subdirectories
real_directory = os.path.join(base_directory, 'real')
fake_directory = os.path.join(base_directory, 'fake')

process_videos_in_directory(real_directory)
process_videos_in_directory(fake_directory)

print("Video processing completed.")
