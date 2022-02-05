import cv2
import numpy as np
import os, os.path
import random

face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

for i in range(140):
    try:
        img = cv2.imread('Modi/{}.jpg'.format(i))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x,y,w,h = face_cascade.detectMultiScale(gray)[0]
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        img = img[y:y+h, x:x+w]
        if random.random() > 0.2:
            cv2.imwrite('./Data/Train/{}.jpg'.format(i),img)
        else:
            cv2.imwrite('./Data/Validation/{}.jpg'.format(i),img)
    except Exception as e:
        print(i)
        print(e)
        pass
