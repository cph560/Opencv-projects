import cv2
import numpy as np
import time 

cam = cv2.VideoCapture(0)
wh = 320
confThreshold = 0.4
nmsThreshold = 0.2
prev_frame_time = 0
new_frame_time = 0


classpath = 'coco.names'
with open(classpath, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelcfg = 'yolov3.cfg'
modelweight = 'yolov3.weights'

nn = cv2.dnn.readNetFromDarknet(modelcfg, modelweight)
nn.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
nn.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def get_objects_no(layer_output, img):
    wt, ht, ct = img.shape
    bbox = []
    confs = []
    objects = []
    for output in layer_output:
        for row in output:
            classids = row[5:]
            ind = np.argmax(classids)
            conf = classids[ind]
            if conf > confThreshold:
                confs.append(float(conf))
                w, h = int(row[2]*wt), int(row[3]*ht)
                x, y = int(row[0]*wt-w/2), int(row[1]*ht-h/2)
                bbox.append([x, y, w, h])
                objects.append(ind)
    indeices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    print(indeices)
    for i in indeices:
        
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, box, (0, 255, 0), 2)
        cv2.putText(img, f'{classes[objects[i]].upper()} {round(confs[i], 2)}', (x ,y+10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0),2)
# print(len(classes))
while True:
    success, img = cam.read()

    blob = cv2.dnn.blobFromImage(img, 1/255, (wh, wh), swapRB=True, crop=False)

    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    cv2.putText(img, str(fps), (7, 70),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (100, 255, 0),2)

    nn.setInput(blob)
    layername = list(nn.getLayerNames())
    # print(len(layername))
    outputlayer = [layername[i-1] for i in nn.getUnconnectedOutLayers()]
    output = nn.forward(outputlayer)
    get_objects_no(output, img)
    cv2.imshow('cam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()