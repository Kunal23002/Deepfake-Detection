# DeepFake Detection Web Application

A state-of-the-art web application that uses an ensemble of deep learning models to detect deepfake videos. The application analyzes videos and provides predictions with confidence scores about whether the content is real or fake.

## Features

- Real-time video analysis
- Ensemble model approach combining multiple deep learning architectures
- Confidence score generation
- User-friendly web interface
- Face detection and analysis
- Support for multiple video formats (mp4, mov, avi)

## Technical Architecture

The application uses an ensemble of four powerful models:

1. **EfficientNetB0**: A lightweight but efficient convolutional neural network
2. **XceptionNet**: A deep convolutional neural network architecture
3. **EfficientNetB7**: A more complex version of EfficientNet for higher accuracy
4. **PPG (PhotoPlethysmoGraphy)**: A novel model that analyzes blood flow patterns in videos

## Requirements

```
torch
torchvision
streamlit
opencv-python
mtcnn
efficientnet-pytorch
timm
numpy
Pillow
matplotlib
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deepfake-detection-webapp.git
cd deepfake-detection-webapp
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the model weights and place them in the appropriate directory:
- efficientnet-b0.pth
- xception-net.pth
- efficientnet-b7.pth
- ppg.pth
- ensemble_model.pth

## Usage

1. Start the Streamlit application:
```bash
streamlit run DeepFake_Detection_Web_App.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Upload a video file through the web interface

4. Wait for the analysis to complete

5. View the results, including:
   - Prediction (Real/DeepFake)
   - Confidence score
   - Visual representation of predictions

## Project Structure

```
deepfake-detection-webapp/
├── DeepFake_Detection_Web_App.py    # Main application file
├── requirements.txt                 # Python dependencies
├── uploads/                         # Directory for uploaded videos
├── models/                          # Directory for model weights
│   ├── efficientnet-b0.pth
│   ├── xception-net.pth
│   ├── efficientnet-b7.pth
│   ├── ppg.pth
│   └── ensemble_model.pth
└── README.md                        # Project documentation
```

## Model Weights

The model weights are not included in the repository due to their size. You'll need to:
1. Train the models separately
2. Download pre-trained weights
3. Place them in the `models/` directory

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any queries or support, please contact:
- Email: kuvalkgarg@gmail.com
- GitHub: [kuvalkgarg](https://github.com/kuvalkgarg)
- LinkedIn: [kuvalkgarg](https://www.linkedin.com/in/kuvalkgarg/) 