import streamlit as st
from model import EfficientNetB0, XceptionNet, EfficientNetB7, Attention_mask, DeepPhys, EnsembleModel
import timm
import cv2
import io
import numpy as np
import matplotlib.pyplot as plt
from preprocess import video_processor

efficientnetb0_model = EfficientNetB0(num_classes=2)
# Load the saved weights into the model
efficientnetb0_model.load_state_dict(efficientnetb0_weights, strict=False)

# Create an instance of the model
xceptionnet_model = timm.create_model('xception', pretrained=True, num_classes=2)
xceptionnet_model.load_state_dict(xceptionnet_weights, strict=False)

efficientnetb7_model = EfficientNetB7(num_classes=2)
# Load the saved weights into the model
efficientnetb7_model.load_state_dict(efficientnetb7_weights, strict=False)

if list(ppg_weights.keys())[0].startswith('module.'):
    ppg_weights = {k[7:]: v for k, v in ppg_weights.items()}

# Initialize the model with the same parameters as the original model
ppg_model = DeepPhys(in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3,
                         dropout_rate1=0.8, dropout_rate2=0.8, pool_size=(2, 2), nb_dense=128, img_size=72)

# Load the matched state_dict into the model
ppg_model.load_state_dict(ppg_weights)

ensemble_model = EnsembleModel(efficientnetb0_model, xceptionnet_model, ppg_model, efficientnetb7_model, [0.10, 0.20, 0.0, 0.70])





st.title("Video Upload and Processing")

# Create a file uploader for video files
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.write("Uploaded video:")
    st.write("Format of video", uploaded_file.type)
    st.video(uploaded_file)

    video_data = uploaded_file.read()  # Read the video data as bytes
    # print(video_data)
    result= video_processor('temp_video.mp4')
    print(len(result[0]))

