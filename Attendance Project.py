import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path= 'ImageAttendance'
images= []
classNames= []
myList=os.listdir(path)
print(myList)

for cl in myList:        # cl= class  cur= current
    curImg= cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])  # this will give name without .jpg
print(classNames)

def findEncodings(images):
    encodeList= []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# to mark attendance

def markAttendance(name):
    with open('Attendance.csv','r+')as f:
        myDataList= f.readlines()
        print(myDataList)
        nameList= []
        for line in myDataList:
            entry= line.split(' , ')
            nameList.append(entry[0])
        if name not in nameList:
            now= datetime.now()
            dtString= now.strftime('%d:%H')
            f.writelines(f'\n{name},{dtString}')



encodeListKnown= findEncodings(images)  # to run this we call function
print('Encoding Complete')

# math the image with web cam

cap= cv2.VideoCapture(0)  # id=0

# while loop to get each frame

while True:          # cap.read= gives us img  imgS= imgsmall  facesCurFrame= current frame
    success, img= cap.read()#pixel size   0.25= scale
    imgS= cv2.resize(img, (0,0), None,   0.25,0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):  #we want all 4 in same loo so we use zip
        matches= face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis= face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex= np.argmin(faceDis)

        if matches[matchIndex]:
            name= classNames[matchIndex].upper()
            y1,x2,y2,x1= faceLoc
            y1, x2, y2, x1= y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1, y1), (x2,y2), (0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0), cv2.FILLED)
            cv2.putText(img, name,(x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)




