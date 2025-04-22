import dlib # type: ignore
import numpy as np
import cv2
import pandas as pd
import time
import logging
import sqlite3
import datetime
import os
import threading


db_lock = threading.Lock()

# Dlib  / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib landmark / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


# Define the attendance table
def create_attendance_table():
    """Create attendance table if it does not exist."""
    with sqlite3.connect("attendance.db", timeout=10) as conn:
        cursor = conn.cursor()
        create_table_sql = f"""CREATE TABLE IF NOT EXISTS attendance (
            employee_id TEXT,
            name TEXT,
            entry_time TEXT,
            exit_time TEXT,
            date DATE,
            UNIQUE(employee_id, date)
        )"""
        cursor.execute(create_table_sql)
        conn.commit()


# Create a connection to the database
db_lock = threading.Lock()

def attendance(self, employee_id, name, camera_name):
    """Logs entry and continuously updates exit time, keeping the last exit detection."""
    with db_lock:
        with sqlite3.connect("attendance.db", timeout=10) as conn:
            cursor = conn.cursor()
            employee_id = str(employee_id)
            today_date = datetime.datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            
            cursor.execute("SELECT entry_time, exit_time FROM attendance WHERE employee_id = ? AND date = ?", (employee_id, today_date))
            result = cursor.fetchone()
            
            if camera_name.lower() == "entry camera":
                if result is None:
                    # Mark entry time if no record exists
                    cursor.execute("INSERT INTO attendance (employee_id, name, date, entry_time) VALUES (?, ?, ?, ?)", (employee_id, name, today_date, current_time))
                    logging.info(f"✅ Entry recorded for {name} at {current_time}.")
                else:
                    logging.info(f"⚠️ {name} already marked present today. Skipping entry.")
            elif camera_name.lower() == "exit camera":
                if result:
                    # Update exit time for each detection (only latest time will be considered as final exit)
                    cursor.execute("UPDATE attendance SET exit_time = ? WHERE employee_id = ? AND date = ?", (current_time, employee_id, today_date))
                    logging.info(f"🔄 Exit time updated for {name}: {current_time}")
            conn.commit()


class Face_Recognizer:
    def __init__(self, rtsp_url, name):
        self.rtsp_url = rtsp_url
        self.name = name
        self.font = cv2.FONT_ITALIC

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # cnt for frame
        self.frame_cnt = 0

        #  Save the features of faces in the database
        self.face_features_known_list = [] # Known face features
        # / Save the name of faces in the database
        self.face_name_known_list = [] # Known names

        #  List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # List to save names of objects in frame N-1 and N
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        #  cnt for faces in frame N-1 and N
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        #  Save the features of people in current frame
        self.current_frame_face_feature_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        #  Reclassify after 'reclassify_interval' frames
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

        self.face_id_known_list = []  # Known employee IDs

    #  "features_all.csv"  / Get known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=0)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_id_known_list.append(csv_rd.iloc[i, 0])
                self.face_name_known_list.append(csv_rd.iloc[i, 1])
                for j in range(2, 130):
                    if csv_rd.iloc[i, j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i, j])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0

    def update_fps(self):
        now = time.time()
        # Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    # / Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1, dtype=np.float64)
        feature_2 = np.array(feature_2, dtype=np.float64)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # / Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            #  For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    #  cv2 window / putText on cv2 window
    def draw_note(self, img_rd):
        #  / Add some info on windows
        cv2.putText(img_rd, "Face Recognizer Mastek", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple(
                [int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]),
                                 self.font,
                                 0.8, (255, 190, 0),
                                 1,
                                 cv2.LINE_AA)


    def attendance(self, employee_id, name, camera_name):
        """Logs entry and continuously updates exit time, keeping the last exit detection."""
        with db_lock:
            with sqlite3.connect("attendance.db", timeout=10) as conn:
                cursor = conn.cursor()
                employee_id = str(employee_id)
                today_date = datetime.datetime.now().strftime("%Y-%m-%d")
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                
                cursor.execute("SELECT entry_time, exit_time FROM attendance WHERE employee_id = ? AND date = ?", (employee_id, today_date))
                result = cursor.fetchone()
                
                if camera_name.lower() == "entry camera":
                    if result is None:
                        # Mark entry time if no record exists
                        cursor.execute("INSERT INTO attendance (employee_id, name, date, entry_time) VALUES (?, ?, ?, ?)", (employee_id, name, today_date, current_time))
                        logging.info(f"✅ Entry recorded for {name} at {current_time}.")
                    else:
                        logging.info(f"⚠️ {name} already marked present today. Skipping entry.")
                elif camera_name.lower() == "exit camera":
                    if result:
                        # Update exit time for each detection (only latest time will be considered as final exit)
                        cursor.execute("UPDATE attendance SET exit_time = ? WHERE employee_id = ? AND date = ?", (current_time, employee_id, today_date))
                        logging.info(f"🔄 Exit time updated for {name}: {current_time}")
                conn.commit()

    #  Face detection and recognition wit OT from input video stream
    def process(self, stream, camera_name):
    # 1. Get known faces from "features.all.csv"
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug(f"[{camera_name}] Frame {self.frame_cnt} starts")

                # Read frame from stream
                flag, img_rd = stream.read()

                if not flag or img_rd is None:
                    print(f"[{camera_name}] Failed to read frame from stream. Exiting...")
                    break  # Exit loop if no frame is received

                img_rd = cv2.resize(img_rd, (720, 560))

                # Face detection
                #faces = detector(img_rd, 0)
                #faces = detector(img_rd, 1)
                faces = detector(img_rd, 1)
                if len(faces) == 0:
                    logging.debug(f"[{camera_name}] No faces detected in this frame.")
                #   continue  # Skip processing for this frame
                


                # Save the current frame uniquely for each camera
                frame_filename = f"frame_{camera_name.lower()}.jpg"
                cv2.imwrite(frame_filename, img_rd)

                # Update frame face count
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                    self.reclassify_interval_cnt != self.reclassify_interval
                ):
                    logging.debug(f"[{camera_name}] No face count changes in this frame.")

                    self.current_frame_face_position_list = []

                    if "unknown" in self.current_frame_face_name_list:
                        self.reclassify_interval_cnt += 1

                    if self.current_frame_face_cnt != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(
                                (faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4))
                            )
                            self.current_frame_face_centroid_list.append(
                                [(faces[k].left() + faces[k].right()) / 2, (faces[k].top() + faces[k].bottom()) / 2]
                            )

                            img_rd = cv2.rectangle(
                                img_rd, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 255), 2
                            )

                        if self.current_frame_face_cnt != 1:
                            self.centroid_tracker()

                        for i in range(self.current_frame_face_cnt):
                            img_rd = cv2.putText(
                                img_rd,
                                self.current_frame_face_name_list[i],
                                self.current_frame_face_position_list[i],
                                self.font,
                                0.8,
                                (0, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
                        self.draw_note(img_rd)

                else:
                    logging.debug(f"[{camera_name}] Face count changes detected.")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt = 0

                    if self.current_frame_face_cnt == 0:
                        logging.debug(f"[{camera_name}] No faces detected.")
                        self.current_frame_face_name_list = []
                    else:
                        logging.debug(f"[{camera_name}] Running face recognition.")
                        self.current_frame_face_name_list = []
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape)
                            )
                            self.current_frame_face_name_list.append("unknown")

                        for k in range(len(faces)):
                            logging.debug(f"[{camera_name}] Processing face {k+1}.")
                            self.current_frame_face_centroid_list.append(
                                [(faces[k].left() + faces[k].right()) / 2, (faces[k].top() + faces[k].bottom()) / 2]
                            )
                            self.current_frame_face_X_e_distance_list = []

                            self.current_frame_face_position_list.append(
                                (faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4))
                            )

                            for i in range(len(self.face_features_known_list)):
                                if str(self.face_features_known_list[i][0]) != "0.0":
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_feature_list[k], self.face_features_known_list[i]
                                    )
                                    logging.debug(f"[{camera_name}] Distance with person {i+1}: {e_distance_tmp}")
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    self.current_frame_face_X_e_distance_list.append(999999999)

                            similar_person_num = self.current_frame_face_X_e_distance_list.index(
                                min(self.current_frame_face_X_e_distance_list)
                            )

                            if min(self.current_frame_face_X_e_distance_list) < 0.8:
                                self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                                logging.debug(
                                    f"[{camera_name}] Recognized: {self.face_name_known_list[similar_person_num]}"
                                )

                                # Insert attendance record
                                employee_id = self.face_id_known_list[similar_person_num]
                                name = self.face_name_known_list[similar_person_num]

                                self.attendance(employee_id, name, camera_name)  # Pass camera_name to log entry/exit

                            else:
                                logging.debug(f"[{camera_name}] Unknown person detected.")

                        self.draw_note(img_rd)

                key = cv2.waitKey(1) & 0xFF

            # Different keys for different cameras
                if camera_name.lower() == "entry camera" and key == ord("e"):  # Press 'e' to close Entry Camera
                    print("Closing Entry Camera...")
                    break  
                elif camera_name.lower() == "exit camera" and key == ord("x"):  # Press 'x' to close Exit Camera
                    print("Closing Exit Camera...")
                    break  


                self.update_fps()
                cv2.imshow(f"{camera_name} Camera", img_rd)

                logging.debug(f"[{camera_name}] Frame processing complete.\n")



    def run(self):
        logging.info(f"Connecting to {self.name}: {self.rtsp_url}...")
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            logging.error(f"Could not open RTSP stream for {self.name}. Check RTSP URL and network connection.")
            return 

        logging.info(f"{self.name} RTSP stream opened successfully.")
        self.process(cap,self.name)

        cap.release()
        cv2.destroyAllWindows()
        logging.info(f"{self.name} RTSP stream closed.")



def main():
    logging.basicConfig(level=logging.INFO)  # Set logging level

    # Create attendance table if it does not exist
    create_attendance_table()

    # RTSP URLs for entry and exit cameras
    entry_camera_url = "rtsp://admin:M8$tek12@192.168.1.250:554/cam/realmonitor?channel=1&subtype=1&proto=onvif-tcp"
    exit_camera_url = "rtsp://admin:Admin@123@192.168.1.251:554/cam/realmonitor?channel=1&subtype=1&proto=onvif-tcp"

    entry_camera = Face_Recognizer(entry_camera_url, "Entry Camera")
    exit_camera = Face_Recognizer(exit_camera_url, "Exit Camera")

    # Run both cameras using threads
    entry_thread = threading.Thread(target=entry_camera.run)
    exit_thread = threading.Thread(target=exit_camera.run)

    entry_thread.start()
    exit_thread.start()

    entry_thread.join()
    exit_thread.join()


if __name__ == '__main__':
    main()
