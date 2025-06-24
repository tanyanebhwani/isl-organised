# 🤟 Sign Language Recognition using Spatiotemporal CNN + MediaPipe

A real-time system that recognizes four basic sign language gestures—**Morning**, **Afternoon**, **Evening**, and **Night**—from video input. It uses **MediaPipe** for landmark extraction and a custom **Spatiotemporal Convolutional Neural Network (CNN)** to learn temporal patterns from hand and body movements.

---

## 🎯 Project Overview

This system enables communication between individuals who use **sign language** and those who do not. The user simply performs the sign using hand gestures, and the system recognizes the sign and displays its meaning on-screen in real-time.

---

## ✅ Goal

Classify videos into **four sign classes**:
- Morning
- Afternoon
- Evening
- Night

By analyzing **temporal body and hand landmarks** extracted from videos.

---

## 🧠 How the Project Works (Step-by-Step)

### 1. **Preprocessing** (`main.ipynb`)
- **a. MediaPipe Landmark Extraction**
  - Converts raw videos into "skeleton videos" using MediaPipe.
  - Stored in: `data/Days_and_Time_skeleton/`
  - Function used: `use_mediapipe()` from `mediapipe/mp.py`

- **b. Frame Extraction + Feature Preparation**
  - Extracts 30 frames per video.
  - Creates NumPy array `X.npy` with shape: `(num_videos, 30, height, width, 3)`
  - Done via `combine_features()` in `feature_extraction/feature_extraction.py`

- **c. Label Assignment**
  - Labelled based on folder names (e.g., Morning, Night, etc.)
  - Saved in `y.npy`

### 2. **Train/Validation/Test Split**
- 80% training + validation, 20% testing
- Further 90/10 split of training into train and val

### 3. **Model Training**
- Defined in `models/model.py`
- Learns **spatial + temporal features** of gesture sequences
- Input shape: `(30, height, width, 3)`
- Evaluation: Accuracy, confusion matrix, classification report
- Output: `my_sign_model.h5`

### 4. **Real-Time Inference** (`userInterface.py`)
- Uses webcam for live frame capture
- Applies MediaPipe and builds buffer of 30 frames
- Passes data to model and displays predicted gesture live

---

## 💻 Project Requirements

### 🔧 Hardware
- A working **Webcam**

### 🧰 Software
- **OS**: Windows 8 or later
- **IDE**: Jupyter Notebook
- **Language**: Python 3.9

### 📦 Python Libraries
- `opencv-python`
- `numpy`
- `tensorflow`
- `keras`
- `mediapipe`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

## 📦 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/tanyanebhwani/isl-organised
cd isl-organised
```

### 2. Install Dependencies
```bash
pip install numpy scikit-learn seaborn matplotlib opencv-python mediapipe tensorflow
```
### 3. Run the System
If you're using Jupyter Notebook, run:

```bash
%run ./userInterface.py
```
### 4. Using the System
Use the spacebar to record your sign.

The predicted sign (Morning, Night, etc.) will be displayed in the top right corner of the screen.

📁 Folder Structure

## isl-organised/
│
├── main.ipynb                   # Main notebook for training

├── userInterface.py             # Real-time prediction interface

│

├── feature_extraction/

│   └── feature_extraction.py    # Generates training features from skeleton videos

│

├── mediapipe/

│   └── mp.py                    # MediaPipe conversion script

│

├── models/

│   └── model.py                 # Spatiotemporal CNN model definition

│

├── data/

│   ├── Days_and_Time/           # Raw input gesture videos

│   └── Days_and_Time_skeleton/  # Skeletonized videos using MediaPipe

│

├── X.npy                        # Input features (30 frames/video)

├── y.npy                        # Corresponding labels

├── my_sign_model.h5             # Trained model weights

└── README.md                    # Project documentation

## 📊 Model Details
Type: 3D Spatiotemporal CNN

Input: (30, height, width, 3)

Trained On: Skeleton videos of hand/body landmarks

Output: One of four classes (Morning, Afternoon, Evening, Night)

## 🔮 Future Improvements
👥 Increase Vocabulary Size
Expand the model to recognize more sign classes beyond the current four.

🗣️ Text-to-Speech Integration
Convert the predicted sign into spoken output, enabling two-way communication.

📱 Mobile App Version
Deploy the model using TensorFlow Lite or MediaPipe on Android/iOS.

🧠 Model Optimization
Reduce model size and improve inference speed for real-time performance on low-end devices.

🖐️ Two-Hand Sign Support
Extend recognition to include signs that involve both hands or facial expressions.

🌐 Web App Deployment
Build a browser-based interface using WebRTC + TensorFlow.js for real-time sign recognition online.


## 🤝 Contact
For any questions, suggestions, or collaboration inquiries:

📧 Email: nebhwanitanya0@gmail.com

💻 GitHub: @tanyanebhwani

