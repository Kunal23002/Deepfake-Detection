import os
import cv2
import torch
from mtcnn import MTCNN
import streamlit as st
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import matplotlib.pyplot as plt
import base64
import imageio
from io import BytesIO
import numpy as np
from efficientnet_pytorch import EfficientNet
import timm

# Define a folder to store uploaded videos
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

video_file = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize variables to count predictions and calculate confidence score
class_counts = {}
confidence = 0

# Loading trained models with their weights
efficientnetb0_weights = torch.load(
    r"path/to/efficientnet-b0.pth",
    map_location=torch.device(device),
)
xceptionnet_weights = torch.load(
    r"path/to/xception-net.pth",
    map_location=torch.device(device),
)
efficientnetb7_weights = torch.load(
    r"path/to/efficientnet-b7.pth",
    map_location=torch.device(device),
)
ppg_weights = torch.load(
    r"path/to/ppg.pth",
    map_location=torch.device(device),
)

# Define EfficientNetB0 model architecture


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB0, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained(
            "efficientnet-b0", num_classes=num_classes
        )

    def forward(self, x):
        return self.efficientnet(x)


# Create an instance of the model
efficientnetb0_model = EfficientNetB0(num_classes=2)

# Load the saved weights into the model
efficientnetb0_model.load_state_dict(efficientnetb0_weights, strict=False)
efficientnetb0_model.to(device)
efficientnetb0_model.eval()

# Define XceptionNet model architecture


class XceptionNet(nn.Module):
    def __init__(self, num_classes):
        super(XceptionNet, self).__init__()
        self.xceptionnet = timm.create_model("xception", pretrained=True)

        # Modify the output layer for binary classification
        num_ftrs = self.xceptionnet.fc.in_features
        self.xceptionnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.xceptionnet(x)


# Create an instance of the model
xceptionnet_model = XceptionNet(num_classes=2)

# Load the saved weights into the model
xceptionnet_model.load_state_dict(xceptionnet_weights, strict=False)
xceptionnet_model.to(device)
xceptionnet_model.eval()

# Define EfficientNetB7 model architecture


class EfficientNetB7(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB7, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained(
            "efficientnet-b7", num_classes=num_classes
        )

    def forward(self, x):
        return self.efficientnet(x)


# Create an instance of the model
efficientnetb7_model = EfficientNetB7(num_classes=2)

# Load the saved weights into the model
efficientnetb7_model.load_state_dict(efficientnetb7_weights, strict=False)
efficientnetb7_model.to(device)
efficientnetb7_model.eval()

# Define PPG model architecture


class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        """May be generated manually."""
        config = super(Attention_mask, self).get_config()
        return config


class DeepPhys(nn.Module):
    def __init__(
        self,
        in_channels=3,
        nb_filters1=32,
        nb_filters2=64,
        kernel_size=3,
        dropout_rate1=0.25,
        dropout_rate2=0.5,
        pool_size=(2, 2),
        nb_dense=128,
        img_size=36,
    ):
        """Definition of DeepPhys.
        Args:
        in_channels: the number of input channel. Default: 3
        img_size: height/width of each frame. Default: 36.
        Returns:
        DeepPhys model.
        """
        super(DeepPhys, self).__init__()

        # self.final_dense_1 = nn.Linear(16384, 128, bias=True)

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense

        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(
            self.in_channels,
            self.nb_filters1,
            kernel_size=self.kernel_size,
            padding=(1, 1),
            bias=True,
        )
        self.motion_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True
        )
        self.motion_conv3 = nn.Conv2d(
            self.nb_filters1,
            self.nb_filters2,
            kernel_size=self.kernel_size,
            padding=(1, 1),
            bias=True,
        )
        self.motion_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True
        )

        # Apperance branch convs
        self.apperance_conv1 = nn.Conv2d(
            self.in_channels,
            self.nb_filters1,
            kernel_size=self.kernel_size,
            padding=(1, 1),
            bias=True,
        )
        self.apperance_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True
        )
        self.apperance_conv3 = nn.Conv2d(
            self.nb_filters1,
            self.nb_filters2,
            kernel_size=self.kernel_size,
            padding=(1, 1),
            bias=True,
        )
        self.apperance_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True
        )

        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(
            self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True
        )
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(
            self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True
        )
        self.attn_mask_2 = Attention_mask()

        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)

        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)

        # Dense layers
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        else:
            raise Exception("Unsupported image size")

        # Final dense layer with a single neuron
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)

    def forward(self, inputs, params=None):
        diff_input = inputs[:, :3, :, :]
        raw_input = diff_input

        d1 = torch.tanh(self.motion_conv1(diff_input))
        d2 = torch.tanh(self.motion_conv2(d1))

        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))

        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        d5 = torch.tanh(self.motion_conv3(d4))
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        # Clone and detach to keep in the computation graph
        out = torch.sigmoid(self.final_dense_2(d11))
        return out


# In case the keys have the 'module.' prefix, remove it to match the keys in the current model
if list(ppg_weights.keys())[0].startswith("module."):
    ppg_weights = {k[7:]: v for k, v in ppg_weights.items()}

# Initialize the model with the same parameters as the original model
ppg_model = DeepPhys(
    in_channels=3,
    nb_filters1=32,
    nb_filters2=64,
    kernel_size=3,
    dropout_rate1=0.8,
    dropout_rate2=0.8,
    pool_size=(2, 2),
    nb_dense=128,
    img_size=72,
)

# Load the matched state_dict into the model
ppg_model.load_state_dict(ppg_weights)

ppg_model.to(device)
ppg_model.eval()


class EnsembleModel(nn.Module):
    def __init__(self, model1, model2, model3, model4, weights):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.weights = weights

    def forward(self, x):
        preds1 = self.model1(x)
        preds2 = self.model2(x)
        preds3 = self.model3(x)
        preds4 = self.model4(x)

        # Apply softmax to convert logits to probabilities
        preds1_probs = torch.softmax(preds1, dim=1)
        preds2_probs = torch.softmax(preds2, dim=1)
        preds3_probs = torch.softmax(preds3, dim=1)
        preds4_probs = torch.softmax(preds4, dim=1)

        # Calculate the ensemble prediction (weighted average)
        weighted_predictions = (
            (preds1_probs * self.weights[0])
            + (preds2_probs * self.weights[1])
            + (preds3_probs * self.weights[2])
            + (preds4_probs * self.weights[3])
        )

        return weighted_predictions


# Load your ensemble model
ensemble_model = EnsembleModel(
    efficientnetb0_model,
    xceptionnet_model,
    ppg_model,
    efficientnetb7_model,
    [0.10, 0.20, 0.00, 0.70],
)
ensemble_model.load_state_dict(
    torch.load(
        r"path/to/ensemble_model.pth",
        map_location=torch.device("cpu"),
    )
)
ensemble_model.eval()

# Transformation for frames from videos
transform = transforms.Compose(
    [
        transforms.Resize(72),
        transforms.CenterCrop(72),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Function to extract frames and predict


def extract_frames_and_predict(video_path):
    class_counts = {}
    # Prediction code

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Parameters for frame extraction
    frame_interval = 10  # Extract a frame every 10 frames (adjust as needed)
    max_frames = 300  # Maximum number of frames to extract

    frame_count = 0

    # Initialize the MTCNN detector
    mtcnn = MTCNN()

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Detect faces in the frame using MTCNN
            faces = mtcnn.detect_faces(frame)

            if faces:
                # Get the first detected face
                x, y, w, h = faces[0]["box"]
                face_img = frame[y : y + h, x : x + w]

                # Convert the face image to a format compatible with the model
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(face_rgb)

                # Preprocess the frame and get the model's prediction
                pil_frame = transform(pil_frame).unsqueeze(0)
                with torch.no_grad():
                    output = ensemble_model(pil_frame)

                    # Get the predicted class
                    predicted_class = output.argmax().item()

                    # Update class counts
                    if predicted_class not in class_counts:
                        class_counts[predicted_class] = 0
                    class_counts[predicted_class] += 1

        frame_count += 1

    # Calculate the sum of all values in the dictionary
    total_count = sum(class_counts.values())

    # Check if the value of key '1' is greater than or equal to 15% of the total count
    if class_counts.get(1, 0) >= 0.15 * total_count:
        final_prediction = 1
        if class_counts[0] > class_counts[1]:
            confidence = class_counts[final_prediction] / sum(class_counts.values())
            confidence = 100 - (confidence * 100)
        else:
            confidence = class_counts[final_prediction] / sum(class_counts.values())
            confidence = confidence * 100

    else:
        final_prediction = 0
        confidence = class_counts[final_prediction] / sum(class_counts.values())
        confidence = confidence * 100

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()

    # Swap the dictionary values only if final_prediction == 1
    if final_prediction == 1 and class_counts.get(0, 0) > class_counts.get(1, 0):
        class_counts[0], class_counts[1] = class_counts[1], class_counts[0]

    # Return the result
    if final_prediction == 0:
        result = (
            "Prediction: Real, Confidence Score: "
            + str(confidence)
            + "%. "
            + "This metric is a probability score that tells you how confident I am that my model has extracted the correct value.",
            class_counts,
        )
    else:
        result = (
            "Prediction: DeepFake, Confidence Score: "
            + str(confidence)
            + "%. "
            + "This metric is a probability score that tells you how confident I am that my model has extracted the correct value.",
            class_counts,
        )

    return result


# Streamlit App

# Create a Streamlit sidebar for navigation buttons
st.sidebar.markdown(
    """
    <style>
    .sidebar-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    .stButton > button,
    .stFileUploader > div > div > div > label {
        display: block;
        width: 100%;
        text-align: center;
        padding: 10px;
        margin: 5px 0;
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton > button:hover,
    .stFileUploader > div > div > div > label:hover {
        background-color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Use session state to store the page state
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Add buttons for Home, Scan, and Contact
if st.sidebar.button("Home"):
    st.session_state.page = "Home"

if st.sidebar.button("Scan"):
    st.session_state.page = "Scan"

if st.sidebar.button("Contact"):
    st.session_state.page = "Contact"

if st.session_state.page == "Home":
    st.title("DeepFake Detection Web App")
    # Define the text content
    text_content = [
        "Deepfake videos are a disturbing and rapidly emerging technological phenomenon that has the potential to pose serious threats to individuals, society, and even national security. The term 'deepfake' is a portmanteau of 'deep learning' and 'fake', and it refers to the use of artificial intelligence(AI) techniques, particularly deep learning algorithms, to create hyper-realistic forged videos, audio recordings, or images. These creations convincingly depict people saying or doing things that they never did, often with malicious intent",
        "One of the most notable examples of deepfake misuse is the 2019 'deepfake' of Mark Zuckerberg, the CEO of Facebook. The video, which depicted Zuckerberg delivering a scripted speech about the power of Facebook, was so convincing that it raised significant concerns about the potential misuse of deepfake technology in manipulating public opinion and spreading false information.",
        "My web application makes use of an ensemble approach in order to classify whether your submitted video is a deepfake or real. This state-of-the-art model consists of 3 extremely powerful models, namely, EfficientNetB0, XceptionNet and EfficientNetB7. Along with this, I've incorporated a novel PhotoPlethysmoGraphy model as well, which analyses the volumetric change in blood flow in the video submitted by you, and helps detect the real video through this. My final prediction follows a majority voting scheme and provides a confidence score along with the prediction.",
    ]

    # Define the image file paths (local)
    image_paths = [
        r"path/to/image1.jpg",
        r"path/to/image2.jpg",
        r"path/to/image3.jpg",
    ]

    # Create a Streamlit layout with alternating text and image
    for i in range(len(text_content)):
        cols = st.columns(2)
        text = text_content[i]
        image_path = image_paths[i]

        if i % 2 == 0:
            cols[0].write(text)  # Left column for text
            cols[1].image(
                Image.open(image_path), use_column_width=True
            )  # Right column for image
        else:
            cols[1].write(text)  # Right column for text
            cols[0].image(
                Image.open(image_path), use_column_width=True
            )  # Left column for image


if st.session_state.page == "Scan":
    st.title("Scan a Video Below")
    if "uploaded_video" not in st.session_state:
        st.session_state.uploaded_video = None

    if st.session_state.uploaded_video is None:
        st.session_state.uploaded_video = st.file_uploader(
            "Upload a Video", type=["mp4", "mov", "avi"]
        )

    if st.session_state.uploaded_video is not None:
        video_path = os.path.join(UPLOAD_FOLDER, st.session_state.uploaded_video.name)
        with open(video_path, "wb") as video_file:
            video_file.write(st.session_state.uploaded_video.read())

        # Display the video
        video_file = open(video_path, "rb").read()

    if st.session_state.uploaded_video is not None:
        st.markdown("<h2>Results</h2>", unsafe_allow_html=True)
        with st.spinner("Analysing..."):
            prediction, class_counts = extract_frames_and_predict(video_path)
            st.write(f"{prediction}")

            # Display the bar graph in the main content area
            if (
                class_counts
            ):  # To make sure the graph doesn't render before the prediction
                # Create a figure and axis for the graph
                fig, ax = plt.subplots()

                # Set a custom color palette for the bars
                colors = ["#000000", "#ff0000"]

                # Define class labels
                class_labels = ["Real", "DeepFake"]

                # Create bar positions
                bar_positions = [0, 1]  # Set positions for "Real" and "DeepFake"

                # Create the bar chart for values greater than 0
                bar_heights = [class_counts.get(label, 0) for label in bar_positions]

                # Create the bar chart
                ax.bar(bar_positions, bar_heights, color=colors, alpha=0.7)

                # Add data labels to the bars
                for x, count in zip(bar_positions, bar_heights):
                    ax.text(x, count, str(count), ha="center", va="bottom", fontsize=12)

                # Customize the appearance
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_linewidth(0.5)
                ax.spines["bottom"].set_linewidth(0.5)
                ax.yaxis.set_ticks_position("left")
                ax.xaxis.set_ticks_position("bottom")

                # Set labels and title
                ax.set_xlabel("Class")
                ax.set_ylabel("Count")
                ax.set_title("Predictions")

                # Set x-axis labels
                ax.set_xticks(bar_positions)
                ax.set_xticklabels(class_labels)

                # Display the bar graph to the right of the video
                col1, col2 = st.columns(2)
                col1.video(video_file)
                col2.pyplot(fig)

if st.session_state.page == "Contact":
    st.title("Developer Information")
    # Contact details (text) with new lines and bigger font size
    contact_text = """
    Name: Kuval Kush Garg and Kunal E 
    """

    # Right image
    right_image = Image.open(
        r"path/to/image.jpg"
    )

    # Create a Streamlit layout with text on the left and image on the right
    cols = st.columns([2, 1])  # Create a 2-column layout

    # Add text on the left column with custom CSS for styling
    cols[0].markdown(
        f'<p style="font-size: 20px;">\n{contact_text}\n```</p>', unsafe_allow_html=True
    )

    # Add the image on the right column
    cols[1].image(right_image, use_column_width=True)


# Run the Streamlit app
if __name__ == "__main__":
    st.set_option("deprecation.showfileUploaderEncoding", False)
