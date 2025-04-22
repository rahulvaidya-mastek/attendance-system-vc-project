import os
import dlib  # type: ignore
import csv
import numpy as np
import logging
import cv2
import sqlite3

# Function to load face database from SQLite
def load_face_db():
    conn = sqlite3.connect("face_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT emp_id, name FROM employees")

    face_db = {}
    for emp_id, name in cursor.fetchall():
        if isinstance(emp_id, bytes):  # Convert binary to string
            emp_id = int.from_bytes(emp_id, byteorder='big')  # Convert bytes to integer
        face_db[str(emp_id)] = name  # Store as a string
    
    conn.close()
    return face_db

# Path of cropped faces
path_images_from_camera = "data/data_faces_from_camera/"

# Use Dlib's frontal face detector
detector = dlib.get_frontal_face_detector()

# Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Function to return 128D features for a single image
def return_128d_features(path_img):
    img_rd = cv2.imread(path_img)
    faces = detector(img_rd, 1)

    logging.info("Processing image: %s", path_img)

    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = np.zeros(128)  # Default empty vector if no face detected
        logging.warning("No face detected in %s", path_img)

    return face_descriptor

# Function to return the mean value of 128D face descriptor for a person
def return_features_mean_personX(path_face_personX):
    features_list_personX = []
    
    if not os.path.exists(path_face_personX):
        logging.warning("Folder not found: %s", path_face_personX)
        return np.zeros(128)

    photos_list = os.listdir(path_face_personX)
    
    if photos_list:
        for photo in photos_list:
            logging.info("Reading image: %s", os.path.join(path_face_personX, photo))
            features_128d = return_128d_features(os.path.join(path_face_personX, photo))
            if np.any(features_128d):  # Only add valid features
                features_list_personX.append(features_128d)
    else:
        logging.warning("No images found in: %s", path_face_personX)

    return np.mean(features_list_personX, axis=0) if features_list_personX else np.zeros(128)

def main():
    logging.basicConfig(level=logging.INFO)

    # Load employee data from database
    face_db = load_face_db()

    person_list = sorted(os.listdir(path_images_from_camera))

    with open("data/features_all.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["Employee_ID", "Name"] + [f"Feature_{i}" for i in range(1, 129)]
        writer.writerow(header)

        for person in person_list:
            person_folder = os.path.join(path_images_from_camera, person)
            features_mean_personX = return_features_mean_personX(person_folder)

            parts = person.split('_')
            emp_id = parts[2] if len(parts) >= 3 else "Unknown"

            # Fetch name from face_db
            emp_name = face_db.get(emp_id, "Unknown")

            features_row = [str(emp_id), str(emp_name)] + features_mean_personX.astype(str).tolist()
            writer.writerow(features_row)
            logging.info("Extracted features for: %s (ID: %s, Name: %s)", person, emp_id, emp_name)

    logging.info("Saved all extracted features to: data/features_all.csv")

if __name__ == '__main__':
    main()
