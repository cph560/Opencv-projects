import cv2
import numpy as np
import face_recognition
import os 
from datetime import datetime

def findencode(images):
    encodlis = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodlis.append(encode)
    return encodlis

def markattendece(path, name):
    with open(path, 'r+') as f:
        data = f.readlines()
        namelist = []
        for line in data:
            line = line.split(',')
            namelist.append(line[0])
        if name not in namelist:
            now = datetime.now()
            time = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{time}')

if __name__ =='__main__':
    
    path = 'Attendence'

    listpath = os.listdir(path)
    image = []
    classes = []

    for cls in listpath:
        img = cv2.imread(f'{path}/{cls}')
        image.append(img)
        classes.append(os.path.splitext(cls)[0])


    encodelisknown = findencode(image)
    print('Encoding Complete')

    cam = cv2.VideoCapture(0)

    while True:
        success, img = cam.read()
        img_resize = cv2.resize(img, (0,0), None, 0.25, 0.25)
        img_cvt = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

        faces_locs = face_recognition.face_locations(img_cvt)
        encodefaces = face_recognition.face_encodings(img_cvt)
        for encode, loc in zip(encodefaces, faces_locs):
            match = face_recognition.compare_faces(encodelisknown, encode)
            faceDis = face_recognition.face_distance(encodelisknown,encode)
            
            ind = np.argmin(faceDis)

            if match[ind]:
                name = classes[ind].upper()
                y1, x2, y2, x1 = faces_locs
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255, 0),2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255, 0),2)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2)
                markattendece('attendence.csv', name)
                print(name)
        
        cv2.imshow('Camera', img)
        cv2.waitKey(1)