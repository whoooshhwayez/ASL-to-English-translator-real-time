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

print("=== Sophisticated ASL Data Collector ===")
print("Press 's' to start collecting for a letter")
print("Press SPACE to capture one image")
print("Press 'q' to quit")

cap = cv2.VideoCapture(0)

for letter in letters:
    print(f"\n--- Collecting data for letter: {letter} ---")
    print("Press 's' to start capturing 100 images for this letter")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Letter: {letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 's' to start", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        cv2.imshow('Collect ASL Data', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            print(f"\nStarting collection for {letter}...")
            break
        elif key == ord('q'):
            print("Quitting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    count = 0
    captured_images = []  # Track images for this letter
    
    while count < 100:
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            # Extract landmarks
            landmarks = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Normalize relative to wrist
            wrist_x, wrist_y = landmarks[0], landmarks[1]
            for i in range(0, len(landmarks), 3):
                landmarks[i] -= wrist_x
                landmarks[i+1] -= wrist_y

            # Show capture status
            cv2.putText(frame, f"Capturing {count+1}/100", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS
            )
            
            # Wait for SPACE to capture
            cv2.imshow('Collect ASL Data', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                # Add label and save
                row = landmarks + [letter]
                data.append(row)
                count += 1
                print(f"Captured image {count} for letter {letter}")
                
        else:
            cv2.putText(frame, "Show hand clearly", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Collect ASL Data', frame)
            key = cv2.waitKey(1) & 0xFF
            
        if key == ord('q'):
            print("Quitting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    print(f"✅ Completed {letter}: {count} images captured")

# Save to CSV
df = pd.DataFrame(data)
df.to_csv('asl_data/asl_landmarks_sophisticated.csv', index=False)
print(f"\n🎉 Collected {len(data)} samples. Saved to asl_data/asl_landmarks_sophisticated.csv")

cap.release()
cv2.destroyAllWindows()
