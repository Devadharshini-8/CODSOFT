import face_recognition
import cv2
import os
import numpy as np

# Load known images
known_faces_dir = "known_faces"
known_encodings = []
known_names = []

print("[INFO] Loading known faces...")
for filename in os.listdir(known_faces_dir):
    img_path = os.path.join(known_faces_dir, filename)
    img = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(img)
    if encoding:
        known_encodings.append(encoding[0])
        known_names.append(os.path.splitext(filename)[0])

print("[INFO] Starting face recognition...")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Webcam feed started. Press 'q' to quit.")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read from webcam.")
        break

    print("Running frame...")  # Debug print

    # Resize frame to 1/4 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]

        # Scale back up face locations
        top, right, bottom, left = face_location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the video frame with boxes
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quitting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
