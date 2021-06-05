import cv2
import numpy as np
import time

#pretrained data/images/weights
weights=r'C:/.../darknet-master/yolov3.weights'
#architecture
cfg=r'C:/.../darknet-master/cfg/yolov3.cfg'
net = cv2.dnn.readNet(weights, cfg)
#net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg") #Tiny Yolo

classes = []
with open(r"C:/.../darknet-master/data/coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)

colors= np.random.uniform(0,255,size=(len(classes),3))
font = cv2.FONT_HERSHEY_PLAIN

layer_names = net.getLayerNames()

#net.getUnconnectedOutLayers() are == array([[200],[227],[254]])
#outputlayers = [layer_names[200 - 1] for i in net.getUnconnectedOutLayers()]
#**************
#since index starts from 0 in python i.e. in 'layer_names' 
#and from in net.getUnconnectedOutLayers() 
#**************
#layer_names[199]= 'yolo_82'
#layer_names[226]= 'yolo_94'
#layer_names[253]= 'yolo_106'

outputlayers=[]
for i in net.getUnconnectedOutLayers():
    outputlayers.append(layer_names[i[0] - 1])
outputlayers

#copy-paste the downloaded deepsort folder to location : C:\Users\SHRIKAR DESAI
from deep_sort.tracker import Tracker

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r'C:\...\Driving.mp4')
starting_time= time.time()
frame_id = 0

while True:
    b,frame= cap.read() # 
    if(b == True and frame is not None):
        frame_id+=1
        frame = cv2.resize(frame,(800,500))
        height,width,channels = frame.shape
        #detecting objects
        blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop=False) #reduce 416 to 320    
        #blob = cv2.dnn.blobFromImage(image, scalefactor:(1/255)=0.00392, size, bgr(grey), swapRGB ot BGR, crop) 
            
        net.setInput(blob)
        #by giving net.forward() we get o/p from o/p layers
        # outs detects number of faces
        outs = net.forward(outputlayers)
        #len(outs)
        #print(outs[2])
        
        #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
        class_ids=[]
        confidences=[]
        boxes=[]
        #outs=[[0.1,0.,....,0.],[0.2,0.,....,0.],[0.3,0.,....,0.]]   in out 3 list are there normally
        #1st out i.e.[0.1,0.,....,0.] 
        #len(detection)
        #len(detection[5:])
        #from 5th num in list there are confidances of class_id and as in line 'scores = detection[5:]'
        #length of detection is 85 last 80 positions are of class_id in the order given in classes, so np.argmax gives index            which has max value (and not the actual max value)
        
        #detection=[x,y,width,height,unknown,[5-80]are class_id]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    #object detected
                    #x and y are centre of obj detected
                    center_x= int(detection[0]*width)
                    center_y= int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
                    
                    #rectangle co-ordinaters
                    x=int(center_x - w/2)
                    y=int(center_y - h/2)
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
                    boxes.append([x,y,w,h]) #put all rectangle areas
                    confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                    class_ids.append(class_id) #name of the object tha was detected
        
        #cv2.dnn.NMSBoxes(boxes,confidences,score_threshold,IOU_threshold) 
        #if 1 obj is detected multiple times so [IOU= (Area of overlap / area of union) of all boxes] and in o/p 1 box is shown
        indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
        count=0
        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                lb=['bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat']
                #lb=['bottle']
                if label in lb:
                    count+=1
                    label = label+' '+str(count)
                    confidence= confidences[i]
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                    cv2.putText(frame,label+" "+str(round(confidence,2)),(x+20,y+30),font,1,(255,255,255),2)
                   #cv2.putText(frame,label+" "+str(round(confidence,number of points after decimal)),(x,y-10),font,1,(0,0,255),2)
    
                
        elapsed_time = time.time() - starting_time
        fps=frame_id/elapsed_time
        cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1)
        
        cv2.imshow("Image",frame)
        key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
        
        if key == ord('q'): #esc key stops the process
            break;
    else:
        break
    
cv2.destroyAllWindows()
cap.release()  


















































































