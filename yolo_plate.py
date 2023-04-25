import cv2 as cv
import os
import time
import matplotlib.pyplot as plt
import easyocr
Conf_threshold = 0.2
NMS_threshold = 0.2
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]
this_path=os.getcwd()
class_name = []
with open('classes.names', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
print(class_name)

#get the yolo model weights & params
net = cv.dnn.readNet('model.weights', 'darknet-yolov3.cfg') #Deep Neural Network module
# read net : Read deep learning network represented in one of the supported formats. 

#net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA) #Ask network to use specific computation backend where it supported.
#net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16) #Ask network to make computations on specific target device.


model = cv.dnn_DetectionModel(net) #This class represents high-level API for object detection networks.
#DetectionModel allows to set params for preprocessing input image. DetectionModel creates net from file with trained weights and config, sets preprocessing input,
#runs forward pass and return result detections. For DetectionModel SSD, Faster R-CNN, YOLO topologies are supported.



model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
#reader=easyocr.Reader(['en'])
'''
img=cv.imread('test_private/910.jpg')
w,h,_=img.shape

blob = cv2.dnn.blobFromImage(image=img, scale=1/255, size=(320, 320))

net.SetInput(blob)

detections=utlis.get_outputs(net)
print(detections)
'''

img_path='test_private/910.jpg'
for i in range(1113,1114):
    frame = cv.imread(f'test_private/{i}.jpg')
    #H,W,_=frame.shape
    #print(H,W)

    #_, frame = cap.read()
    #print(frame.shape)
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        color=(0,0,255)
        label=class_name[classid]
        print(box)
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
            
        cv.rectangle(frame, [left,top,width,height], color, 1)
        #cv.putText(frame, label, (box[0], box[1]-10),
        #                   cv.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cropped=frame[int(top):int(top+height),int(left):int(left+width)]
        #cv.circle(frame, (left,top), 3, color, -1)
        #print(reader.readtext(cropped,detail=0))



        cv.imwrite(f'result_{i}.png',cropped)
    cv.imshow('frame',frame)
    #cap=cv.VideoCapture(0)
    #get the dimensions for saving the video
'''
#frame_width = int(cap.get(3))
#frame_height = int(cap.get(4))
#size = (frame_width, frame_height)
#print(size)

'''




while True:
    ret, frame = cap.read()
    #frame=cv.resize(frame, (700, 600))

    if ret == False:
        break
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        #color = COLORS[int(classid) % len(COLORS)]
        color=(0,0,255)
        #label = "%s : %f" %(class_name[classid], score)
        label=class_name[classid]
        print(box)
        
        cv.rectangle(frame, box, color, 3)
        cv.putText(frame, label, (box[0], box[1]-10),
                       cv.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
   

    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

'''
