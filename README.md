# Regional_Sign_Language_Recognition_Tool_for_Hindi_and_Kannada-

A real-time sign language recognition system for Hindi and Kannada languages using computer vision and machine learning techniques.

## 🚀 Project Overview

This project implements a comprehensive sign language recognition system that can detect and translate hand gestures into speech and text in real-time. The system focuses on Hindi and Kannada sign languages, making it accessible to regional communities.

**Project Duration:** June 2024 – August 2024

## ✨ Features

- **Real-time Recognition:** Instant hand gesture detection and classification
- **Multi-language Support:** Hindi and Kannada sign language recognition
- **Speech Synthesis:** Converts recognized gestures to audio using text-to-speech
- **Skeletal Hand Tracking:** Robust 21-point hand landmark detection
- **Low Latency:** Optimized for real-time performance
- **Multiple Processing Modes:** Grayscale, binary, and skeletal representations

## 🛠️ Technologies Used

- **Computer Vision:** OpenCV, CVZone
- **Machine Learning and Training:** CNN (Convolutional Neural Networks)
- **Speech Synthesis:** OpenAI TTS / Google Text-to-Speech (GTTS)
- **Programming Language:** Python
- **Hand Tracking:** MediaPipe (via CVZone)

## 🏗️ System Architecture

### Core Components

1. **Hand Detection Module** (`HandDetector`)
   - Detects up to 1 hand in real-time
   - Extracts 21 hand landmarks
   - Provides bounding box coordinates


     ![Skeleton_Image_1718864726 663953](https://github.com/user-attachments/assets/847586a3-c8e9-4627-991a-e98c42a312ed)


3. **Image Processing Pipeline**
   - Grayscale conversion
   - Adaptive thresholding
   - Binary image generation
   - Skeletal structure creation

4. **Classification Module**
   - CNN-based gesture recognition
   - Trained on 8 classes: `["ka", "kha", "ga", "gha", "ca", "cha", "ja", "jha"]`
   - Real-time prediction with confidence scoring

5. **Text-to-Speech Integration**
   - OpenAI TTS for high-quality speech synthesis
   - Automatic audio playback for recognized gestures

## 📁 Project Structure

```
sign-language-recognition/
├── datacollection.py          # Data collection script
├── high5.py                   # Advanced preprocessing pipeline
├── test4.py                   # Main recognition system
├── model/
│   ├── keras_model(8cls).h5   # Trained CNN model
│   └── labels(8cls).txt       # Class labels
├── data(new)/                 # Training data directory
└── README.md
```

## 🔧 Installation & Setup

### Prerequisites

```bash
pip install opencv-python
pip install cvzone
pip install numpy
pip install tensorflow
pip install openai
pip install pygame  # For audio playback
```

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd sign-language-recognition
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure OpenAI API**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

4. **Prepare Model Files**
   - Ensure `keras_model(8cls).h5` and `labels(8cls).txt` are in the `model/` directory

## 🎯 Usage

### Data Collection
Run the data collection script to gather training samples:
```bash
python datacollection.py
```
- Press `s` to save images
- Adjust the `folder` variable for different classes

### Training Data Preprocessing
Process collected images with advanced preprocessing:
```bash
python high5.py
```
- Generates skeletal representations
- Creates binary and grayscale versions
- Press `s` to save processed images

### Real-time Recognition
Start the main recognition system:
```bash
python test4.py
```
- Show hand gestures to the camera
- System will recognize and speak the gesture after 10 seconds of consistent detection
- Press `q` to quit

## 🎨 Image Processing Pipeline

The system employs a sophisticated multi-stage processing approach:

1. **Hand Detection & Cropping**
   - Detects hand using MediaPipe
   - Extracts region of interest with padding

2. **Preprocessing**
   - Grayscale conversion
   - Gaussian blur for noise reduction
   - Adaptive thresholding for binary images

3. **Skeletal Representation**
   - 21-point hand landmark extraction
   - Connected skeletal structure drawing
   - Normalized positioning for consistency

4. **Classification**
   - CNN-based prediction on skeletal images
   - Confidence-based gesture recognition

## 📊 Model Details

- **Architecture:** Convolutional Neural Network
- **Training Platform:** Google Teachable Machine
- **Classes:** 8 Hindi/Kannada characters
- **Input Size:** 400x400 pixels (skeletal images)
- **Output:** Gesture classification with confidence scores

## 🔄 Workflow

1. **Capture:** Real-time video feed from camera
2. **Detect:** Hand detection and landmark extraction
3. **Process:** Multi-stage image preprocessing
4. **Classify:** CNN-based gesture recognition
5. **Synthesize:** Text-to-speech conversion
6. **Output:** Audio playback of recognized gesture

## 🎛️ Configuration

### Adjustable Parameters

```python
# Hand detection settings
maxHands = 1
offset = 20
imgsize = 300

# Recognition settings
print_duration = 10  # seconds for consistent detection
```

### Model Customization

- Modify `labels` array to add new gestures
- Retrain model with additional classes
- Update preprocessing pipeline as needed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 Future Enhancements

- [ ] Support for more regional languages
- [ ] Mobile app development
- [ ] Real-time sentence formation
- [ ] Gesture sequence recognition
- [ ] Multi-hand gesture support
- [ ] Web-based interface

## 🐛 Known Issues

- Lighting conditions may affect recognition accuracy
- Hand positioning needs to be within camera frame
- Background complexity can impact detection

## 🙏 Acknowledgments

- CVZone for hand tracking utilities
- Google Teachable Machine for model training platform
- OpenAI for text-to-speech services
- MediaPipe for hand landmark detection



**Note:** This project aims to bridge communication gaps and make sign language more accessible to broader communities. Your contributions and feedback are welcome!
