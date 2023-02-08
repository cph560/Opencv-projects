import cv2
import numpy as np

cam = cv2.VideoCapture(0)


classpath = 'coco.names'
with open(classpath, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

print(len(classes))
while True:
    success, img = cam.read()
    cv2.imshow('cam', img)
    cv2.waitKey(1)