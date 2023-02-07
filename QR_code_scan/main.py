import cv2
import numpy as np
from pyzbar.pyzbar import decode
'''Test Part'''
img = cv2.imread('1.PNG')
code = decode(img)
print(code)
pts = np.array([code[0].polygon], np.int32)
pts = pts.reshape(-1, 1, 2)
pt2 = code[0].rect
text = code[0].data.decode('utf-8')
cv2.polylines(img, [pts], True, (255,0,255), 5)
cv2.putText(img, text, (pt2[0], pt2[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1)
cv2.imshow('img',img)
cv2.waitKey(0)

'''Camera Part'''
cam = cv2.VideoCapture(0)
cam.set(3, 640) #width
cam.set(4, 480) #height

while True:
    
    success, img =  cam.read()
    for info in decode(img):
        mydata = info.data.decode('utf-8')
        rect = info.rect
        ptsI = np.array([info.polygon], np.int32)
        pts = ptsI.reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, (255,0,255), 5)
        cv2.putText(img, mydata, (rect[0], rect[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1)
        # print(mydata, rect)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()