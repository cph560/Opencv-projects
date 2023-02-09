import cv2 
import numpy as np
import time 
# img = cv2.imread('lena.png')

confthres = 0.5
coeNMS = 0.2 #the coe smaller, NMS(non-maximum suppression) stronger
cam = cv2.VideoCapture(0)
cam.set(3, 1280) #width
cam.set(4, 720) #height
cam.set(10,150) #brightness
prev_frame_time = 0
new_frame_time = 0

classpath = './Object_Detection_Files/coco.names'
with open(classpath, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
config_path = './Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weight_path = './Object_Detection_Files/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weight_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True:
    success, img = cam.read()
    Ids, confs, bbox = net.detect(img, confThreshold=confthres)
    bbox_list = list(bbox)
    conf_list = list(np.array(confs).reshape(1, -1)[0])
    conf_list = list(map(float, conf_list))

    indices = cv2.dnn.NMSBoxes(bbox_list, confs, confthres, coeNMS)
    # print(Ids, confs)
    # print(bbox_list)
    # print(indices)
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    cv2.putText(img, str(fps), (7, 70),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (100, 255, 0),2)
                    
    for i in indices:
        box = bbox_list[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=3)
        cv2.putText(img, classes[Ids[i]-1], (x, y+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, str(round(confs[i], 2)), (x, y+h-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    
    
    # if len(Ids)!=0:
    #     for id, confidence, box in zip(Ids.flatten(), confs.flatten(), bbox):
    #         cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
            
    #         # cv2.putText(img, classes[id-1]+f'  probability:{str(round(confidence, 2))}', (bbox[0][0],bbox[0][1]-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    #         cv2.putText(img, classes[id-1], (bbox[0][0], bbox[0][1]+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    #         cv2.putText(img, str(round(confidence, 2)), (bbox[0][0], bbox[0][1]+bbox[0][3]-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    # print(classes)

    cv2.imshow('img', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()