import face_recognition
import cv2
import os
import numpy as np

# Load known faces
known_faces_dir = "known_faces"
known_encodings = []
known_names = []

print("[INFO] Loading known faces from:", known_faces_dir)
for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])
        else:
            print(f"[WARNING] No face found in {filename}, skipping.")

if not known_encodings:
    print("[ERROR] No known faces loaded. Please add a clear photo.")
    exit()

print("[INFO] Known faces loaded successfully.")

# Load the uploaded video
video_path = "unknown_faces/test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"[ERROR] Could not open video file: {video_path}")
    exit()

print("[INFO] Starting face recognition on video...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video.")
        break

    # Resize and convert for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for encoding, location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, encoding)
        best_match = np.argmin(face_distances)

        if matches[best_match]:
            name = known_names[best_match]

        # Scale back to original size
        top, right, bottom, left = [v * 4 for v in location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("Face Recognition - Uploaded Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()
