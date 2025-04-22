from flask import Flask, render_template, request, send_file  # type: ignore
import sqlite3
import csv
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', selected_date='', no_data=False)

@app.route('/attendance', methods=['POST'])
def attendance():
    selected_date = request.form.get('selected_date')
    selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    formatted_date = selected_date_obj.strftime('%Y-%m-%d')

    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    cursor.execute("SELECT employee_id, name, entry_time, exit_time FROM attendance WHERE date = ?", (formatted_date,))
    attendance_data = []
    for emp_id, name, entry_time, exit_time in cursor.fetchall():
        emp_id = int.from_bytes(emp_id, "big") if isinstance(emp_id, bytes) else emp_id
        total_time = calculate_total_time(entry_time, exit_time)
        attendance_data.append((emp_id, name, entry_time, exit_time, total_time))
    
    conn.close()

    if not attendance_data:
        return render_template('index.html', selected_date=selected_date, no_data=True)

    return render_template('index.html', selected_date=selected_date, attendance_data=attendance_data)

@app.route('/download_attendance', methods=['POST'])
def download_attendance():
    selected_date = request.form.get('selected_date')
    selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    formatted_date = selected_date_obj.strftime('%Y-%m-%d')
    
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute("SELECT employee_id, name, entry_time, exit_time FROM attendance WHERE date = ?", (formatted_date,))
    attendance_data = [(emp_id, name, entry_time, exit_time, calculate_total_time(entry_time, exit_time)) for emp_id, name, entry_time, exit_time in cursor.fetchall()]
    conn.close()

    if not attendance_data:
        return render_template('index.html', selected_date=selected_date, no_data=True)
    
    filename = f"attendance_{formatted_date}.csv"
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Employee ID", "Name", "Entry Time", "Exit Time", "Total Time Spent"])
        writer.writerows(attendance_data)
    
    return send_file(filename, as_attachment=True)

def calculate_total_time(entry_time, exit_time):
    if entry_time and exit_time:
        entry_time_obj = datetime.strptime(entry_time, '%H:%M:%S')
        exit_time_obj = datetime.strptime(exit_time, '%H:%M:%S')
        total_time = exit_time_obj - entry_time_obj
        return str(total_time)
    return "N/A"

if __name__ == '__main__':
    app.run(debug=True)
