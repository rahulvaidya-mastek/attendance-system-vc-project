import cv2
import numpy as np
import time


class BehavioralAnalytics:
    def __init__(self, video_source):
        self.cap = cv2.VideoCapture(video_source)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.object_tracks = {}
        self.last_seen = {}
        self.loitering_time = {}
        self.abandoned_threshold = 30  # seconds
        self.loitering_threshold = 20  # seconds
        self.missing_object_threshold = 10  # seconds
    
    def detect_objects(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def track_objects(self, contours, frame):
        current_objects = {}
        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            obj_id = f"{x}-{y}-{w}-{h}"
            current_objects[obj_id] = (x, y, w, h)
            if obj_id not in self.object_tracks:
                self.object_tracks[obj_id] = time.time()
                self.loitering_time[obj_id] = time.time()
            else:
                self.last_seen[obj_id] = time.time()
        return current_objects
    
    def detect_abandoned_objects(self, current_objects):
        for obj_id, start_time in self.object_tracks.items():
            if obj_id in current_objects:
                continue
            if time.time() - start_time > self.abandoned_threshold:
                print(f"⚠️ Abandoned object detected: {obj_id}")
    
    def detect_loitering(self, current_objects):
        for obj_id, start_time in self.loitering_time.items():
            if obj_id in current_objects and time.time() - start_time > self.loitering_threshold:
                print(f"⚠️ Loitering detected for object: {obj_id}")
    
    def detect_missing_objects(self):
        for obj_id, last_seen_time in self.last_seen.items():
            if time.time() - last_seen_time > self.missing_object_threshold:
                print(f"⚠️ Missing object detected: {obj_id}")
                del self.last_seen[obj_id]
    
    def process_frame(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            contours = self.detect_objects(frame)
            current_objects = self.track_objects(contours, frame)
            self.detect_abandoned_objects(current_objects)
            self.detect_loitering(current_objects)
            self.detect_missing_objects()
            for (x, y, w, h) in current_objects.values():
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Behavioral Analytics", frame)
            if cv2.waitKey(30) & 0xFF == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows()

        

if __name__ == "__main__":
    analytics = BehavioralAnalytics("rtsp://admin:M8$tek12@192.168.1.250:554/cam/realmonitor?channel=1&subtype=1&proto=onvif-tcp")
    analytics.process_frame()
