import cv2
import pytesseract


def img2word_single_box(img, color=(0,0,255), thickness=2):
    contents = pytesseract.image_to_boxes(img)
    for i, j in enumerate(contents.splitlines()):
        j = j.split(' ')
        x,y,w,h = int(j[1]), int(j[2]), int(j[3]), int(j[4])
        cv2.rectangle(img, (x,imgH-y),(w,imgH-h),(0,0,255),1)
        cv2.putText(img,j[0],(x,imgH-y+8),cv2.FONT_HERSHEY_COMPLEX,1,color,thickness)
    cv2.imshow('res', img)
    cv2.waitKey(0)

def img2word_whole_word_box(img, color=(0,0,255), thickness=2):
    contents = pytesseract.image_to_data(img)
    
    for i in contents.splitlines()[1:]:
        info = i.split()
        if len(info) == 12:
            x, y, w, h = int(info[6]), int(info[7]), int(info[8]), int(info[9])
            text = info[-1]
            cv2.rectangle(img, (x, y),(w+x, h+y),(0,0,255),1)
            cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,color,thickness)
    cv2.imshow('res', img)
    cv2.waitKey(0)

if __name__ == "__main__":
    
    tesseract_path = 'D:\\Tesseract-OCR\\tesseract.exe'
    
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    # print(pytesseract.get_tesseract_version())
    img = cv2.imread('1.JPG')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgH,imgW, _ = img.shape
    img2word_whole_word_box(img)

