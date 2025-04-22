import dlib # type: ignore
import numpy as np
import cv2
import os
import shutil
import time
import logging
import sqlite3
import tkinter as tk
from tkinter import messagebox, font as tkFont
from PIL import Image, ImageTk
from tkinter import ttk

detector = dlib.get_frontal_face_detector()

class Face_Register:
    def __init__(self):
        self.current_frame_faces_cnt = 0
        self.existing_faces_cnt = 0
        self.ss_cnt = 0
        #os.makedirs(self.path_photos_from_camera, exist_ok=True)

        self.win = tk.Tk()
        self.win.title("Face Register")
        self.win.geometry("1050x560")

        # GUI layout
        self.frame_left_camera = tk.Frame(self.win)
        self.label = tk.Label(self.frame_left_camera)
        self.label.pack()
        self.frame_left_camera.pack(side=tk.LEFT, padx=10, pady=10)

        self.frame_right_info = tk.Frame(self.win)
        self.label_cnt_face_in_database = tk.Label(self.frame_right_info, text=str(self.existing_faces_cnt))
        self.label_fps_info = tk.Label(self.frame_right_info, text="")

        self.input_emp_id = tk.Entry(self.frame_right_info, width=25)
        self.input_name = tk.Entry(self.frame_right_info, width=25)
        self.input_department = tk.Entry(self.frame_right_info, width=25)
        self.input_designation = tk.Entry(self.frame_right_info, width=25)
        self.input_phone = tk.Entry(self.frame_right_info, width=25)

        self.label_warning = tk.Label(self.frame_right_info)
        self.label_face_cnt = tk.Label(self.frame_right_info, text="Faces in current frame: ")
        self.log_all = tk.Label(self.frame_right_info)

        self.font_title = tkFont.Font(family='Helvetica', size=15, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')
        self.font_warning = tkFont.Font(family='Helvetica', size=15, weight='bold')

        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_face_dir = ""

        # Current frame and face ROI position
        self.current_frame = np.ndarray
        self.face_ROI_image = np.ndarray
        self.face_ROI_width_start = 0
        self.face_ROI_height_start = 0
        self.face_ROI_width = 0
        self.face_ROI_height = 0
        self.ww = 0
        self.hh = 0

        self.out_of_range_flag = False
        self.face_folder_created_flag = False

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        self.cap = cv2.VideoCapture(0)  # Get video stream from camera

        # self.cap = cv2.VideoCapture("test.mp4")   # Input local video        
        self.init_db()
        self.GUI_info()
        #self.process()

    #  Delete old face folders
    def GUI_clear_data(self):
        #  "/data_faces_from_camera/person_x/"...
        folders_rd = os.listdir(self.path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(self.path_photos_from_camera + folders_rd[i])
        if os.path.isfile("data/features_all.csv"):
            os.remove("data/features_all.csv")
        self.label_cnt_face_in_database['text'] = "0"
        self.existing_faces_cnt = 0
        self.log_all["text"] = "Face images and `features_all.csv` removed!"

    def GUI_get_input_name(self):
        self.input_name_char = self.input_name.get()
        #self.create_face_folder()
        #self.label_cnt_face_in_database['text'] = str(self.existing_faces_cnt)

    def GUI_get_input_emp_id(self):
        self.input_emp_id_char = self.input_emp_id.get()
        self.create_face_folder()
        self.label_cnt_face_in_database['text'] = str(self.existing_faces_cnt)

    def GUI_get_input_department(self):
        self.input_department_char = self.input_department.get()

    def GUI_get_input_designation(self):
        self.input_designation_char = self.input_designation.get()

    def GUI_get_input_phone(self):
        self.input_phone_char = self.input_phone.get()

    def init_db(self):
        conn = sqlite3.connect("face_data.db")
        cursor = conn.cursor()

    # Create the 'employees' table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                emp_id TEXT,
                name TEXT NOT NULL,
                department TEXT,
                designation TEXT,
                phone TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def save_to_db(self):
        # Fetch inputs
        self.GUI_get_input_emp_id()
        self.GUI_get_input_name()
        self.GUI_get_input_department()
        self.GUI_get_input_designation()
        self.GUI_get_input_phone()

        conn = sqlite3.connect("face_data.db")
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO employees (emp_id, name, department, designation, phone) VALUES (?, ?, ?, ?, ?)", 
                        (self.input_emp_id_char, self.input_name_char, self.input_department_char, self.input_designation_char, self.input_phone_char))
            conn.commit()
            messagebox.showinfo("Success", "Employee registered successfully!")
        except sqlite3.IntegrityError:
            messagebox.showerror("Error", "Employee ID already exists!")
        finally:
            conn.close()

    def GUI_info(self):

        tk.Label(self.frame_right_info,
                 text="Face Hello Mastek ",
                 font=self.font_title).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=2, pady=10)    

        # Step 1: Clear old data
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 1: Clear face photos").grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)
        tk.Button(self.frame_right_info,
                  text='Clear',
                  command=self.GUI_clear_data).grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        
        tk.Label(self.frame_right_info, text="FPS: ").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_fps_info.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)  # Changed from row=1 to row=3

        tk.Label(self.frame_right_info, text="Faces in database: ").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_cnt_face_in_database.grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info,
                 text="Faces in current frame: ").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_face_cnt.grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)

        self.label_warning.grid(row=7, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info, text="Step 2: Face Register", font=self.font_title).grid(row=8, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)

        tk.Label(self.frame_right_info, text="Employee ID:").grid(row=9, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_emp_id.grid(row=9, column=1, padx=5, pady=2)  # Changed row from 1 to 9

        tk.Label(self.frame_right_info, text="Name:").grid(row=10, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_name.grid(row=10, column=1, padx=5, pady=2)  # Changed row from 2 to 10

        tk.Label(self.frame_right_info, text="Department:").grid(row=11, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_department.grid(row=11, column=1, padx=5, pady=2)  # Changed row from 3 to 11

        tk.Label(self.frame_right_info, text="Designation:").grid(row=12, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_designation.grid(row=12, column=1, padx=5, pady=2)  # Changed row from 4 to 12

        tk.Label(self.frame_right_info, text="Phone Number:").grid(row=13, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_phone.grid(row=13, column=1, padx=5, pady=2)  # Changed row from 5 to 13

        tk.Button(self.frame_right_info, text='Save Data', command=self.save_to_db).grid(row=14, column=0, columnspan=2, pady=10)
        self.frame_right_info.pack(side=tk.RIGHT, padx=10, pady=5)

        tk.Button(self.frame_right_info, text='Input', command=self.GUI_get_input_emp_id).grid(row=14, column=1, columnspan=2, padx=10),
        self.frame_right_info.pack(side=tk.RIGHT, padx=10, pady=5)        

        # Step 3: Save current face in frame
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 3: Save face image").grid(row=15, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)

        tk.Button(self.frame_right_info,
                  text='Save current face',
                  command=self.save_current_face).grid(row=16, column=0, columnspan=2, sticky=tk.W, pady=0)

        # Show log in GUI
        self.log_all.grid(row=17, column=0, columnspan=2, sticky=tk.W, padx=0, pady=1)

        self.frame_right_info.pack()        

    def process(self):
        def update_frame():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (480, 360))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(image=img)
                self.label.img_tk = img_tk
                self.label.configure(image=img_tk)

            # Update FPS on every frame
            self.update_fps()   

        self.win.after(20, update_frame)

        update_frame()

    def save_to_db(self):
        emp_id = self.input_emp_id.get()
        name = self.input_name.get()
        department = self.input_department.get()
        designation = self.input_designation.get()
        phone = self.input_phone.get()

        if not emp_id.isdigit() or not (1 <= len(emp_id) <= 10):
            messagebox.showerror("Error", "Employee ID must be between 1 and 10 digits.")
            return
        if len(department) > 20 or len(designation) > 20:
            messagebox.showerror("Error", "Department and Designation must be at most 20 characters.")
            return
        if not phone.isdigit() or len(phone) != 10:
            messagebox.showerror("Error", "Phone number must be exactly 10 digits.")
            return
        
        conn = sqlite3.connect("face_data.db")
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO employees (name, emp_id, department, designation, phone) VALUES (?, ?, ?, ?, ?)",
                           (name, emp_id, department, designation, phone))
            conn.commit()
            messagebox.showinfo("Success", "Data saved successfully!")
        except sqlite3.IntegrityError:
            messagebox.showerror("Error", "Employee ID already exists.")
        finally:
            conn.close()


    # Mkdir for saving photos and csv
    def pre_work_mkdir(self):
        # Create folders to save face images and csv
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

    def get_employee_id(self, person_name):
        """Fetch employee_id from the database using the person's name."""
        conn = sqlite3.connect("face_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT emp_id FROM employees WHERE emp_id, name = ?", (emp_id,  person_name,)) # type: ignore
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None


    # Start from person_x+1
    def check_existing_faces_cnt(self):
        #if os.listdir("data/data_faces_from_camera/"):
            # Get the order of latest person
        #    person_list = os.listdir("data/data_faces_from_camera/")
        #    person_num_list = []
        #    for person in person_list:
        #        person_order = person.split('_')[1].split('_')[0]
        #        person_num_list.append(int(person_order))
        #    self.existing_faces_cnt = max(person_num_list)



        """Ensure folders are created using employee IDs, not person_X_<name>."""
        if os.listdir("data/data_faces_from_camera/"):
            person_list = os.listdir("data/data_faces_from_camera/")
            employee_ids = [int(folder) for folder in person_list if folder.isdigit()]
            self.existing_faces_cnt = max(employee_ids) if employee_ids else 0
        else:
            self.existing_faces_cnt = 0


        # Start from person_1
        #else:
        #    self.existing_faces_cnt = 0


    def register_new_face(self, person_name):
        """Register a new face and store it under the corresponding employee_id."""
        employee_id = self.get_employee_id(person_name)

        if not employee_id:
            print(f"Error: Employee ID not found for {person_name}")
            return

        person_folder = os.path.join(self.path_photos_from_camera, str(employee_id))

        if not os.path.exists(person_folder):
            os.mkdir(person_folder)

        print(f"Face images for {person_name} will be stored in: {person_folder}")



    # Update FPS of Video stream
    def update_fps(self):
        now = time.time()
        #  Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now
        self.label_fps_info["text"] = str(self.fps.__round__(2))

    def create_face_folder(self):
        #  Create the folders for saving faces
        self.existing_faces_cnt += 1
        if self.input_emp_id:
            self.current_face_dir = self.path_photos_from_camera + \
                                    "person_" + str(self.existing_faces_cnt) + "_" + \
                                    self.input_emp_id.get()
        else:
            self.current_face_dir = self.path_photos_from_camera + \
                                    "person_" + str(self.existing_faces_cnt)
        os.makedirs(self.current_face_dir)
        self.log_all["text"] = "\"" + self.current_face_dir + "/\" created!"
        logging.info("\n%-40s %s", "Create folders:", self.current_face_dir)

        self.ss_cnt = 0  #  Clear the cnt of screen shots
        self.face_folder_created_flag = True  # Face folder already created

    def save_current_face(self):
        if self.face_folder_created_flag:
            if self.current_frame_faces_cnt == 1:
                if not self.out_of_range_flag:
                    self.ss_cnt += 1
                    #  Create blank image according to the size of face detected
                    self.face_ROI_image = np.zeros((int(self.face_ROI_height * 2), self.face_ROI_width * 2, 3),
                                                   np.uint8)
                    for ii in range(self.face_ROI_height * 2):
                        for jj in range(self.face_ROI_width * 2):
                            self.face_ROI_image[ii][jj] = self.current_frame[self.face_ROI_height_start - self.hh + ii][
                                self.face_ROI_width_start - self.ww + jj]
                    self.log_all["text"] = "\"" + self.current_face_dir + "/img_face_" + str(
                        self.ss_cnt) + ".jpg\"" + " saved!"
                    self.face_ROI_image = cv2.cvtColor(self.face_ROI_image, cv2.COLOR_BGR2RGB)

                    cv2.imwrite(self.current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg", self.face_ROI_image)
                    logging.info("%-40s %s/img_face_%s.jpg", "Save into：",
                                 str(self.current_face_dir), str(self.ss_cnt) + ".jpg")
                else:
                    self.log_all["text"] = "Please do not out of range!"
            else:
                self.log_all["text"] = "No face in current frame!"
        else:
            self.log_all["text"] = "Please run step 2!"


    def get_frame(self):
        try:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                frame = cv2.resize(frame, (640,480))
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("Error: No video input!!!")

    #  Main process of face detection and saving
    def process(self):
        ret, self.current_frame = self.get_frame()
        faces = detector(self.current_frame, 0)
        # Get frame
        if ret:
            self.update_fps()
            self.label_face_cnt["text"] = str(len(faces))
            #  Face detected
            if len(faces) != 0:
                #   Show the ROI of faces
                for k, d in enumerate(faces):
                    self.face_ROI_width_start = d.left()
                    self.face_ROI_height_start = d.top()
                    #  Compute the size of rectangle box
                    self.face_ROI_height = (d.bottom() - d.top())
                    self.face_ROI_width = (d.right() - d.left())
                    self.hh = int(self.face_ROI_height / 2)
                    self.ww = int(self.face_ROI_width / 2)

                    # If the size of ROI > 480x640
                    if (d.right() + self.ww) > 640 or (d.bottom() + self.hh > 480) or (d.left() - self.ww < 0) or (
                            d.top() - self.hh < 0):
                        self.label_warning["text"] = "OUT OF RANGE"
                        self.label_warning['fg'] = 'red'
                        self.out_of_range_flag = True
                        color_rectangle = (255, 0, 0)
                    else:
                        self.out_of_range_flag = False
                        self.label_warning["text"] = ""
                        color_rectangle = (255, 255, 255)
                    self.current_frame = cv2.rectangle(self.current_frame,
                                                       tuple([d.left() - self.ww, d.top() - self.hh]),
                                                       tuple([d.right() + self.ww, d.bottom() + self.hh]),
                                                       color_rectangle, 2)
            self.current_frame_faces_cnt = len(faces)

            # Convert PIL.Image.Image to PIL.Image.PhotoImage
            img_Image = Image.fromarray(self.current_frame)
            img_PhotoImage = ImageTk.PhotoImage(image=img_Image)
            self.label.img_tk = img_PhotoImage
            self.label.configure(image=img_PhotoImage)

        # Refresh frame
        self.win.after(20, self.process)

    def run(self):
        self.pre_work_mkdir()
        self.check_existing_faces_cnt()
        self.GUI_info()
        self.process()
        self.win.mainloop()
       
        

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Face_Register().run()
