import os
from datetime import datetime
import cv2
import face_recognition
import numpy as np
import csv
import streamlit as st

path = 'Faces'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Initialize attendance status
attendance_status = {name: {'present': False, 'absent': False} for name in classNames}


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name, probability):
    if name and name != "Unknown" and name != "" and probability > 0.6 and name not in recognized_names:
        now = datetime.now()
        dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
        try:
            with open(csv_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                if os.path.getsize(csv_file_path) == 0:
                    writer.writerow(['Name', 'Time', 'Probability'])
                writer.writerow([name, dt_string, f"{probability:.2f}"])
                recognized_names.add(name)

                # Mark attendance status
                attendance_status[name]['present'] = True
                attendance_status[name]['absent'] = False
        except Exception as e:
            print(f"An error occurred while marking attendance: {e}")


now = datetime.now()
csv_file_path = now.strftime('%Y-%m-%d_%H-%M-%S') + '_Attendance.csv'
recognized_names = set()
encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

signal_file_path = 'run_signal.txt'


def check_running_signal():
    if os.path.exists(signal_file_path):
        with open(signal_file_path, 'r') as file:
            return file.read().strip() == 'run'
    return False


# Initialize Streamlit app
st.title("Face Recognition Attendance System")

while True:
    if not check_running_signal():
        break

    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.6)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)
        name = "Unknown"
        probability = 1 - faceDis[matchIndex]

        if matches[matchIndex] and probability > 0.6:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f"{name} {probability:.2f}", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (255, 255, 255), 2)
            markAttendance(name, probability)

    # Refresh present and absent roll numbers
    present_roll_numbers = [roll for roll, status in attendance_status.items() if status['present']]
    absent_roll_numbers = [roll for roll, status in attendance_status.items() if status['absent']]

    # Display results using Streamlit
    st.image(img, channels="BGR", caption=f"Present Roll Numbers: {', '.join(present_roll_numbers)}")
    st.text(f"Absent Roll Numbers: {', '.join(absent_roll_numbers)}")

# Release resources
cap.release()
cv2.destroyAllWindows()
