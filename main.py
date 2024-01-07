import os
from datetime import datetime
import cv2
import face_recognition
import numpy as np
import csv

path = 'Faces'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name, probability):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
    try:
        if name not in recognized_names:
            with open(csv_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                if os.path.getsize(csv_file_path) == 0:
                    writer.writerow(['Name', 'Time', 'Probability'])
                writer.writerow([name, dt_string, f"{probability:.2f}"])
                recognized_names.add(name)

            with open("attendance.txt", "w") as file:
                for present_name in recognized_names:
                    file.write(f"{present_name}\n")
    except Exception as e:
        print(f"An error occurred while marking attendance: {e}")



now = datetime.now()
csv_file_path = now.strftime('%Y-%m-%d_%H-%M-%S') + '_Attendance.csv'
recognized_names = set()
encodeListKnown = findEncodings(images)

signal_file_path = 'run_signal.txt'


def check_running_signal():
    if os.path.exists(signal_file_path):
        with open(signal_file_path, 'r') as file:
            return file.read().strip() == 'run'
    return False


cap = cv2.VideoCapture(0)

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

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()