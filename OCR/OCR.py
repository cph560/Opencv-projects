import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd='D:\\Tesseract-OCR\\tesseract.exe'
img = cv2.imread('1.JPG')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgH,imgW,_ = img.shape
# print(pytesseract.image_to_string(img))
# print(pytesseract.image_to_boxes(img))
contents = pytesseract.image_to_boxes(img)
for i, j in enumerate(contents.splitlines()):
    j = j.split(' ')
    x,y,w,h = int(j[1]), int(j[2]), int(j[3]), int(j[4])
    cv2.rectangle(img, (x,imgH-y),(w,imgH-h),(0,0,255),1)
    cv2.putText(img,j[0],(x,imgH-y+8),cv2.FONT_HERSHEY_COMPLEX,0.5,(10,10,255),1)
cv2.imshow('res', img)
cv2.waitKey(0)


