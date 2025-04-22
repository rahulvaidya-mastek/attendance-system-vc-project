import cv2
import dlib  # type: ignore
import numpy as np
import time
import logging
import threading
from ultralytics import YOLO # type: ignore

# Load YOLOv8 model with higher accuracy
yolo_model = YOLO("yolov8m.pt")  # Use 'yolov8m.pt' for better accuracy

# RTSP URL for CP Plus IP Camera
RTSP_URL = "rtsp://admin:M8$tek12@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0&proto=onvif-tcp"

# Set thresholds
CROWD_THRESHOLD = 1  # More than 5 people is a crowd
CONFIDENCE_THRESHOLD = 0.6  # Confidence threshold to filter out low-confidence detections
IOU_THRESHOLD = 0.5  # Intersection-over-Union threshold for better filtering

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def detect_people_yolo(frame, results):
    """Detects people using YOLOv8 with improved accuracy filtering."""
    detected_boxes = yolo_model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
    person_boxes = []
    for r in detected_boxes:
        for box in r.boxes:
            if int(box.cls) == 0 and box.conf[0] > CONFIDENCE_THRESHOLD:  # Class 0 is 'person' in YOLO
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append((x1, y1, x2, y2))
    results["people"] = person_boxes

def monitor_crowd():
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        logging.error("ERROR: Could not open RTSP stream. Check your RTSP URL and network connection.")
        return
    
    cv2.namedWindow("Crowd Monitoring", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Crowd Monitoring", 800, 600)
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            logging.error("Failed to read frame from stream.")
            break
        frame = cv2.resize(frame, (720, 540))
        
        results = {"people": []}
        
        # Create and start YOLO detection thread
        yolo_thread = threading.Thread(target=detect_people_yolo, args=(frame, results))
        yolo_thread.start()
        yolo_thread.join()
        
        people_boxes = results["people"]
        num_people = len(people_boxes)
        
        # Draw YOLO bounding boxes for detected people
        for (x1, y1, x2, y2) in people_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        cv2.putText(frame, f"People Count: {num_people}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        if num_people > CROWD_THRESHOLD:
            logging.warning("ALERT! Overcrowding detected!")
            cv2.rectangle(frame, (10, 10), (720, 100), (0, 0, 255), -1)
            cv2.putText(frame, "ALERT: OVERCROWDING DETECTED!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "PLEASE MAINTAIN SOCIAL DISTANCING!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Crowd Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor_crowd()
