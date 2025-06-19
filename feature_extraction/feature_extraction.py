import cv2
import os
import numpy as np

def extract_video_frames(video_path, num_frames=30, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)

    frames = []
    count = 0

    while len(frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame = cv2.resize(frame, target_size)
            frame = frame / 255.0  # Normalize pixel values
            frames.append(frame)
        count += 1

    cap.release()

    # Pad if fewer frames
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return frames  # shape: (num_frames, H, W, 3)

DATASET_PATH = "data/Days_and_Time_skeleton"
NUM_FRAMES = 30
TARGET_SIZE = (224, 224)

def combine_features():
    X, y = [], []
    class_names = sorted([
    name for name in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, name)) and not name.startswith('.')
    ])
    class_map = {name: name for idx, name in enumerate(class_names)}
    for class_name in class_names:
        class_path = os.path.join(DATASET_PATH, class_name)
        print(class_name)
        for file in os.listdir(class_path):
            if file.endswith(".mp4"):
                print("heloo")
                video_path = os.path.join(class_path, file)
                frames = extract_video_frames(video_path, NUM_FRAMES, TARGET_SIZE)
                X.append(frames)
                y.append(class_map[class_name])

    X = np.array(X)  # shape: (num_videos, num_frames, H, W, 3)
    y = np.array(y)  # shape: (num_videos,)
    print(X.shape)
    np.save("X.npy", X)
    np.save("y.npy", y)

