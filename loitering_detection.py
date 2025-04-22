from ultralytics import YOLO
import cv2
import face_recognition
import numpy as np
import time
import csv

# Configuration
RESOLUTION = (760, 640)  # Increased resolution for larger output
LOITERING_THRESHOLD = 10  # Seconds
STATIONARY_THRESHOLD = 30  # Pixels movement
FACE_MATCH_THRESHOLD = 0.6
FRAME_SKIP = 5  # Process every other frame for performance

model = YOLO("yolov8n.pt", verbose=False)

# Load registered employee faces from CSV
def load_registered_faces():
    known_faces = []
    known_names = []
    with open("data/features_all.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            emp_id = row[0]
            name = row[1]
            features = np.array(row[2:], dtype=np.float32)
            known_faces.append(features)
            known_names.append(name)
    return known_faces, known_names

known_faces, known_names = load_registered_faces()

# Initialize video capture
camera_url = "rtsp://admin:M8$tek12@192.168.1.250:554/cam/realmonitor?channel=1&subtype=1&proto=onvif-tcp"
cap = cv2.VideoCapture(camera_url)
# cap = cv2.VideoCapture(0) # 0 for default webcam

loitering_threshold = 10  # Time (seconds) before marking as loitering
person_timers = {}  # Stores person ID & entry time
person_positions = {}  # Stores last known positions

def is_stationary(prev_position, curr_position, threshold=30):
    """Check if a person remains in the same location (small movement allowed)."""
    return abs(prev_position[0] - curr_position[0]) < threshold and abs(prev_position[1] - curr_position[1]) < threshold

def detect_loitering_and_faces():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to fetch frame!")
            break

        # Resize the frame to the desired resolution
        frame = cv2.resize(frame, RESOLUTION)

        results = model(frame)  # Detect people

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])  # Get class ID
                if cls == 0:  # Class ID 0 = 'person'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                    person_id = f"{x1}_{y1}"  # Temporary ID (better to use tracking)
                    curr_position = ((x1 + x2) // 2, (y1 + y2) // 2)  # Center point

                    # Check if person is stationary
                    if person_id in person_positions and is_stationary(person_positions[person_id], curr_position):
                        if person_id not in person_timers:
                            person_timers[person_id] = time.time()
                        else:
                            time_spent = time.time() - person_timers[person_id]

                            if time_spent > loitering_threshold:
                                # Crop face region
                                face_crop = frame[y1:y2, x1:x2]
                                rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                                face_locations = face_recognition.face_locations(rgb_face)

                                name = "Unknown"
                                if face_locations:
                                    face_encodings = face_recognition.face_encodings(rgb_face, face_locations)
                                    if face_encodings:
                                        face_encoding = face_encodings[0]
                                        distances = face_recognition.face_distance(known_faces, face_encoding)
                                        best_match_index = np.argmin(distances)

                                        if distances[best_match_index] < FACE_MATCH_THRESHOLD:  # Adjust threshold if needed
                                            name = known_names[best_match_index]

                                # Show loitering alert
                                cv2.putText(frame, f"LOITERING ALERT: {name}", (50, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                                if name == "Unknown":
                                    print(f"⚠️ Loitering Detected! Unknown Person ({int(time_spent)} sec)")
                                else:
                                    print(f"✅ Loitering Detected! Employee: {name}")

                    else:
                        # Reset timer if person moves
                        person_timers.pop(person_id, None)

                    # Update last position
                    person_positions[person_id] = curr_position

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Loitering Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_loitering_and_faces()
