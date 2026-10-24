import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Create folder to save data
if not os.path.exists('asl_data'):
    os.makedirs('asl_data')

# Define ASL letters to collect
letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')  # A-Z
data = []

print("Press key for each letter to collect 100 samples per letter")
print("Press 'q' to quit")

cap = cv2.VideoCapture(0)

for letter in letters:
    print(f"\nCollecting data for letter: {letter}")
    print("Press SPACE to start collection...")
    input()  # Wait for Enter

    count = 0
    while count < 100:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            # Extract landmark coordinates
            landmarks = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Normalize to 0-1 (relative to wrist at index 0)
            wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]
            for i in range(0, len(landmarks), 3):
                landmarks[i] -= wrist_x      # x
                landmarks[i+1] -= wrist_y    # y
                # Skip z (or keep as is)

            # Add label and save
            row = landmarks + [letter]
            data.append(row)
            count += 1

            # Visual feedback
            cv2.putText(frame, f'{letter} - {count}/100', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Collect ASL Data', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Save to CSV
df = pd.DataFrame(data)
df.to_csv('asl_data/asl_landmarks.csv', index=False)
print(f"\nCollected {len(data)} samples. Saved to asl_data/asl_landmarks.csv")

cap.release()
cv2.destroyAllWindows()
