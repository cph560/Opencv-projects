import cv2
import numpy as np
import time 

classpath = 'coco.names'
with open(classpath, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

cam = cv2.VideoCapture(0)
cam.set(3, 640) #width
cam.set(4, 640) #height
confThreshold = 0.5
nmsThreshold = 0.2
prev_frame_time = 0
new_frame_time = 0

net = cv2.dnn.readNet('yolov5s.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

def obj_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape
    x_factor = image_width / 640
    y_factor =  image_height / 640

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)
            
            if (classes_scores[class_id] > 0.25):

                confidences.append(confidence)

                class_ids.append(class_id)
                
                w, h = int(row[2]* x_factor), int(row[3]* y_factor)
                x, y = int(row[0]-w/2* x_factor), int(row[1]-h/2* y_factor)

                boxes.append([x, y, w, h])
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, nmsThreshold)
    # print(boxes)
    for i in indexes:
        
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x ,y), (w, h), (0, 255, 0), 2)
        cv2.putText(img, f'{classes[class_ids[i]].upper()} {round(float(confidences[i]), 2)}', (x ,y+10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0),2)

def format_yolov5(source):

    # put the image in square big enough
    col, row, _ = source.shape
    _max = max(col, row)
    resized = np.zeros((_max, _max, 3), np.uint8)
    resized[0:col, 0:row] = source
    
    # resize to 640x640, normalize to [0,1[ and swap Red and Blue channels
    result = cv2.dnn.blobFromImage(resized, 1/255.0, (640, 640), swapRB=True)
    
    return result

while True:
    success, img = cam.read()

    blob = format_yolov5(img)
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    cv2.putText(img, str(fps), (7, 70),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (100, 255, 0),2)
    net.setInput(blob)
    # layername = list(net.getLayerNames())
    # print(layername)
    predictions = net.forward()
    output = predictions[0]
    # print(output.shape)

    obj_detection(img, output)

    cv2.imshow('cam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
     
# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()