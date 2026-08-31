# DETECTIONNOODLE

Sign Language Detection System using Hand Landmarks and Neural Networks

## Project Overview

DETECTIONNOODLE is a comprehensive system for detecting and classifying Thai sign language using:
- MediaPipe Hand Landmarker for real-time hand pose estimation
- MediaPipe Face Landmarker for facial landmark detection
- Random Forest classifier for character and word recognition
- Real-time video processing with OpenCV

## Features

- Real-time hand and face landmark detection
- Character (Thai letter) recognition from hand gestures
- Word recognition from trained word models
- Voice output using Google Text-to-Speech
- Video data collection and processing
- Model training and evaluation
- Automatic eye blink detection for triggering recording

## Project Structure

```
DETECTIONNOODLE/
├── collect_imgs.py           # Collect training images from camera
├── collect_word_videos.py    # Collect word training videos
├── create_dataset.py         # Process collected data into training dataset
├── train_classifier.py       # Train character and word models
├── inference_classifier.py   # Run real-time inference with trained models
├── Video.py                  # Video processing utilities
├── labels.py                 # Label management and pose definitions
├── console_utils.py          # Console output utilities
├── requirements.txt          # Python dependencies
├── train_words/              # Training word data and videos
│   └── <word>/
│       ├── video/           # Training videos for each word
│       └── *.jpg            # Training images for each word
├── data/                    # Character training data
│   ├── 0/                  # Pose 0 images
│   ├── 1/                  # Pose 1 images
│   └── video/              # Training videos
├── model.p                 # Trained character model
├── word_model.p            # Trained word model
├── hand_landmarker.task    # MediaPipe hand landmarker model
├── face_landmarker.task    # MediaPipe face landmarker model
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- MediaPipe hand and face landmarker task files
- Webcam for real-time inference

### Setup

1. Clone this repository:
```bash
git clone https://github.com/BEE5009/DETECTIONNOODLE.git
cd DETECTIONNOODLE
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download MediaPipe task files:
- Download `hand_landmarker.task` from [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- Download `face_landmarker.task` from [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- Place both files in the project root directory

## Usage

### Training Workflow

1. **Collect Images** (Optional - if pre-trained models not available):
```bash
python collect_imgs.py
```

2. **Collect Word Training Videos**:
```bash
python collect_word_videos.py
```
Keys:
- `A/D`: Navigate between words
- `Q`: Record video for current word
- `ESC`: Exit

3. **Create Training Dataset**:
```bash
python create_dataset.py
```

4. **Train Models**:
```bash
python train_classifier.py
```
This trains both character and word recognition models

### Running Inference

```bash
python inference_classifier.py
```

**Controls:**
- `C`: Toggle character detection ON/OFF
- `V`: Toggle word detection ON/OFF
- `E`: Manually record current character
- `W`: Train/retrain word model from current frame
- `R`: Speak the recognized text
- `T`: Delete last character
- Click on text box: Edit current text
- `Q`: Exit application

## Training Data Structure

### Character Training Data (`data/`)
```
data/
├── 0/          # Character 0
│   ├── 1/      # Pose 1 for character 0
│   ├── 2/      # Pose 2 for character 0
│   └── video/
├── 1/          # Character 1
│   └── 1/
└── video/      # Video files for character training
```

### Word Training Data (`train_words/`)
```
train_words/
├── สวัสดี/
│   ├── video/
│   │   ├── clip_0.mp4
│   │   └── clip_1.mp4
│   └── *.jpg
├── ฉัน/
├── ชื่อ/
└── ...
```

## Supported Thai Words

สวัสดี, ฉัน, ชื่อ, คุณ, อะไร, ทำไม, ขอโทษ, เข้าใจ, ไม่, ได้, หิว, กิน, ข้าว, ยัง, ไม่เป็นไร, ป่วย, สบายดี, นอนหลับ, อย่างไร

## Model Performance

- Character Recognition Accuracy: Trained on collected gesture data
- Word Recognition Accuracy: Trained on video frames from `train_words/`

## System Requirements

- RAM: 4GB minimum (8GB recommended)
- GPU: Optional but recommended for faster processing
- Processor: Intel i5/Ryzen 5 or equivalent minimum

## Dependencies

See `requirements.txt` for complete list:
- opencv-python
- mediapipe
- scikit-learn
- numpy
- PIL
- pygame
- gtts

## Troubleshooting

### Camera not opening
```bash
# On Windows, ensure camera is not in use by other applications
# On Linux, check camera permissions:
sudo usermod -a -G video $USER
```

### MediaPipe models not found
Ensure task files are in project root:
- `hand_landmarker.task`
- `face_landmarker.task`

### No training data found
Run `collect_word_videos.py` or `collect_imgs.py` to generate training data

## Contributing

Feel free to fork and submit pull requests for improvements

## License

MIT License

## Author

BEE5009

## Contact

Email: bimana623@gmail.com
GitHub: https://github.com/BEE5009

---

**Note**: Pre-trained models and sample training videos should be downloaded separately and placed in the appropriate directories before running inference.
