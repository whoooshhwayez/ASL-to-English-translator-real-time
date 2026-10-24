import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load trained model
model = joblib.load('asl_model.pkl')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

print("ASL Translator Running... Press 'q' to quit")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    predicted_letter = "?"

    if results.multi_hand_landmarks:
        # Extract landmarks
        landmarks = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])

        # Normalize relative to wrist (index 0)
        wrist_x, wrist_y = landmarks[0], landmarks[1]
        for i in range(0, len(landmarks), 3):
            landmarks[i] -= wrist_x      # x
            landmarks[i+1] -= wrist_y    # y
            # z is kept as-is

        # Predict
        X = np.array(landmarks).reshape(1, -1)
        predicted_letter = model.predict(X)[0]

        # Draw hand landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS
        )

    # Display prediction
    cv2.rectangle(frame, (20, 20), (200, 100), (0, 255, 0), -1)
    cv2.putText(frame, f'{predicted_letter}', (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
    
    cv2.imshow('ASL Translator', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
