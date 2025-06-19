import cv2
import mediapipe as mp

# Initialize MediaPipe Holistic and Drawing
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Input/output video paths
data = ".../data"
def use_mediapipe(dataset = "Days_and_Time"):
    dataPath = os.path.join(data,dataset);
    for className in os.listdir(dataPath):
        classPath = os.path.join(dataPath,className)
        classLabel = {name: name for idx,name in os.listdir(dataPath)}
        for video in os.listdir(classPath):
            # Capture input video
           if(video.endswith(".MOV")):
               input_video_path = os.path.join(classPath,video)
               cap = cv2.VideoCapture(input_video_path)
               width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
               height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
               fps = cap.get(cv2.CAP_PROP_FPS)
               # Video writer
               fourcc = cv2.VideoWriter_fourcc(*'mp4v')
               out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
               # Initialize Holistic model
               with mp_holistic.Holistic(
                   static_image_mode=False,
                   model_complexity=1,
                   enable_segmentation=False,
                   refine_face_landmarks=True,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.5
               ) as holistic:
              while cap.isOpened():
                  ret, frame = cap.read()
                  if not ret:
                      break
                  # Convert BGR to RGB
                  image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                  # Process the frame
                  results = holistic.process(image_rgb)

                  #Draw landmarks on the original BGR frame
                  mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
                  mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                  mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                  mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                  # Write frame to output video
                  out.write(frame)
        print("Video created successfully")     
    print("✅ Holistic video created successfully!")

cap.release()
out.release()

