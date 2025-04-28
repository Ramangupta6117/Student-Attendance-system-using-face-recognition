import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'D:/desktop/attendance system/Photos'
images = []
classNames = []   
myList = os.listdir(path)
print(myList)
for eachImg in myList:
    curImg = cv2.imread(f'{path}/{eachImg}')
    images.append(curImg)
    classNames.append(os.path.splitext(eachImg)[0])
print(classNames)

def verify_images_loaded():
    print(f"\nVerifying loaded images:")
    print(f"Total images found in directory: {len(images)}")
    print(f"Successfully loaded images: {len(images)}")
    print(f"Number of class names: {len(classNames)}")
    print("\nClass names loaded:")
    for name in classNames:
        print(f"- {name}")

verify_images_loaded()

def findEncodings(images):
    encodeList =[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tString = time_now.strftime('%H:%M:%S')
            dString = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tString},{dString}')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS) #get the location of the face in the image
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame) #get the encoding of the face

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame): #zip the face encoding and the face location
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace) #compare the face encoding with the known face encodings
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace) #calculate the distance between the face encoding and the known face encodings
        print(faceDis)
        matchIndex = np.argmin(faceDis) #get the index of the match
        if matches[matchIndex]:
            name = classNames[matchIndex].upper() #get the name of the match
            print(name)
            y1, x2, y2, x1 = faceLoc #get the location of the face
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4 #scale up the location of the face
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) #draw a rectangle around the face
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 250, 0), cv2.FILLED) #draw a rectangle around the name    
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2) #draw the name on the image       

            markAttendance(name)
    cv2.imshow('webcam', img) #show the image
    if cv2.waitKey(10) == 13: #wait for the user to press enter
        break
cap.release() #release the camera
cv2.destroyAllWindows() #close all windows