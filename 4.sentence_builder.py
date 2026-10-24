import cv2
import mediapipe as mp
import joblib
import numpy as np
import time

# Load trained model
model = joblib.load('asl_model.pkl')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

print("=== ASL Sentence Builder ===")
print("Press 'g' to add SPACE to sentence")
print("Press 's' + SPACE simultaneously to capture letter")
print("Press 'c' to clear sentence")
print("Press 'q' to quit")

cap = cv2.VideoCapture(0)

# Variables for sentence building
current_sentence = ""
confidence_threshold = 0.7

def draw_sentence(frame, sentence):
    """Draw the sentence at the top of the frame"""
    cv2.rectangle(frame, (10, 10), (frame.shape[1]-10, 60), (0, 0, 0), -1)
    cv2.putText(frame, sentence[:40], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def draw_instructions(frame):
    """Draw instructions at the bottom"""
    cv2.rectangle(frame, (10, frame.shape[0]-60), (frame.shape[1]-10, frame.shape[0]-10), (0, 0, 0), -1)
    cv2.putText(frame, "G: Space | S+SPACE: Capture | C: Clear | Q: Quit", (20, frame.shape[0]-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# State variables
captured_letter = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    predicted_letter = "?"
    confidence = 0.0

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

        # Predict
        X = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        if confidence > confidence_threshold:
            predicted_letter = prediction

        # Draw hand landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS
        )

    # Draw UI elements
    draw_sentence(frame, current_sentence)
    draw_instructions(frame)
    
    # Show current prediction
    cv2.rectangle(frame, (frame.shape[1]-150, 10), (frame.shape[1]-10, 60), (0, 255, 0), -1)
    cv2.putText(frame, f"{predicted_letter} ({confidence:.2f})", 
                (frame.shape[1]-140, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    
    # G: Add space to sentence
    if key == ord('g'):
        current_sentence += " "
        print(f"Added SPACE to sentence: '{current_sentence}'")
    
    # S + SPACE: Capture current prediction
    elif key == ord('s'):
        if confidence > confidence_threshold:
            captured_letter = predicted_letter
            print(f"Captured letter: {predicted_letter}")
        else:
            print("Low confidence prediction, not capturing")
    
    # SPACE: Add captured letter to sentence
    elif key == ord(' '):
        if captured_letter:
            current_sentence += captured_letter
            print(f"Added to sentence: '{captured_letter}' -> '{current_sentence}'")
            captured_letter = ""
        else:
            print("No letter captured yet - press 's' first")
    
    # C: Clear sentence
    elif key == ord('c'):
        current_sentence = ""
        captured_letter = ""
        print("Sentence cleared")
    
    # Q: Quit
    elif key == ord('q'):
        break

    cv2.imshow('ASL Sentence Builder', frame)

print(f"\nFinal sentence: '{current_sentence}'")
cap.release()
cv2.destroyAllWindows()sentence_builder.py
