import cv2
import math

path = 'acute-angles.jpg'
img = cv2.imread(path)

pointlist = []
'''click mouse'''
def mousepoint(event,x,y,flags,params):
  if event==cv2.EVENT_LBUTTONDOWN:
    size = len(pointlist)
    if size != 0 and size % 3 != 0:
      cv2.line(img, tuple(pointlist[round((size-1)/3)*3]), (x, y), (0, 0, 255), 3)
    cv2.circle(img, (x,y),5,(0.0,255),cv2.FILLED)
    pointlist.append([x,y])
    print(pointlist)

def calcangle(points):
  sets = points[-3:]
  x = [i[0] for i in sets]
  y = [i[1] for i in sets]
  m1 = (y[1]-y[0])/(x[1]-x[0])
  m2 = (y[2]-y[0])/(x[2]-x[0])
  ang = math.atan((m2-m1)/(1+m1*m2))
  ang = round(math.degrees(ang))
  cv2.putText(img, str(ang), (x[0]+20, y[0]+20), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,255), 2)
  

cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


while True:

  if len(pointlist)%3 == 0 and len(pointlist) != 0:
    calcangle(pointlist)

  cv2.imshow('Image', img)
  cv2.setMouseCallback('Image', mousepoint)
  if cv2.waitKey(1) & 0xFF==ord('q'):
    pointlist = []
    img = cv2.imread(path)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)