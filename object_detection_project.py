#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from threading import Thread
import random
from queue import Queue
import cv2
import pandas as pd
import numpy as np
import time
# from google.colab.patches import cv2_imshow
from keras.preprocessing import image
from matplotlib import pyplot
from PIL import Image
import glob
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split


# In[ ]:


#Take user input for the choice of query the user wants to run - Q1 for only Object Detection and Q2 for Object detection followed by Attribute Classifcation 
query_choice = input("Enter the query you want to run: Q1: For running the object detection & Q2: For running Atttribute Classification: ")
if query_choice == 'Q2': # If the choice is query 2 then load the classifier model
  model_classifier = tf.keras.models.load_model('class_model_updated')
  print("Reloaded Classification model...")


# In[ ]:


#Load YOLO weights and configurations
net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg") #Tiny Yolo
classes = []
with open("coco_classes.txt","r") as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors= np.random.uniform(0,255,size=(len(classes),3))

# Create an empty dataframe to store the predicted counts of cars and classes of cars.
df = pd.DataFrame()

#Assign class 0 to sedan and 1 to suv as the output of classification will be in [0,1] so need to convert them to class names
class_typ ={ 0 : "sedan", 1: "SUV"}

#Read the video using cv2
cap=cv2.VideoCapture("assignment_clip.mp4")
font = cv2.FONT_HERSHEY_PLAIN
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
OUTPUT='assignment2.avi'
#Object for writng the frames to video
writer = cv2.VideoWriter(OUTPUT, fourcc, 30, (428, 576), True)

frame_id = 0
frame_id_lst = []
car_count = []
sedan_list,suv_list = [],[]
fps_list1=[]
fps_list2=[]
time_1=[]
time_2=[]


# In[ ]:


#Create an empty queue for Producer consumer thread for optimizing the computer vision pipeline
queue = Queue()
property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
total_frames = int(cv2.VideoCapture.get(cap, property_id))

#Producer Thread - It stores all the read frames into queue for further processing
class ProducerThread(Thread):
    def run(self):
        frame_id = 0
        global queue
        # Run while loop to read all frames from video
        while frame_id < total_frames:
          _, frame = cap.read()  # Read each frame from video object
          frame_id += 1 # To keep tract of frame number being processed
          data = {"frame_id": frame_id, "frame": frame} # Store the frame number and frame in dictionary
          queue.put(data) # Put the dictionary in queue which will further notify the consumer that queue has some elements present to be processed
          print("Produced", frame_id)
# Consumer Thread            
class ConsumerThread(Thread):
    def run(self):
        frame_id = 0
        global queue
        while frame_id < total_frames :
          data = queue.get() # Read the dictionary of frame number and frame from queue once consumer queue is notified
          frame = data['frame']
          frame_id = data['frame_id']
          queue.task_done()
          height, width, channels = frame.shape # Extract the frame information
          frame_id_lst.append(frame_id)
          # detecting objects in blob by passing arguments like frame, image size, scalefactor
          blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True,
                                        crop=False)  # reduce 416 to 320 to reduce quality of image and reduce space and time complexity

          net.setInput(blob) # pass the above blob or objects in the input layer of the network
          outs = net.forward(outputlayers) # Holds all the information related to extract the event or object detected like x,y coordinate, height and width of the object

          # Showing info on screen/ get confidence score of algorithm in detecting an object in blob
          class_ids = []
          confidences = []
          boxes = []
          sedan_count, suv_count = 0, 0

          for out in outs:
              for detection in out:
                  scores = detection[5:] # For each object get the scores for each class
                  class_id = np.argmax(scores) #Get the class_id with highest score
                  confidence = scores[class_id] # get score or confidence level of the most significant class of the object
                  if (confidence > 0.3 and class_id == 2): #If the model is 30% confident that the object is car then wewill extract the object 
                      # object detected
                      center_x = int(detection[0] * width)
                      center_y = int(detection[1] * height)
                      w = int(detection[2] * width)
                      h = int(detection[3] * height)

                      # cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                      # co-ordinates for the rectangle box
                      x = int(center_x - w / 2)
                      y = int(center_y - h / 2)
                      # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                      boxes.append([x, y, w, h])  # put all rectangle areas

                      confidences.append(
                          float(confidence))  # how confidence was that object detected and show that percentage
                      class_ids.append(class_id)  # name of the object tha was detected
          indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6) # Non Max Suppression for taking only one unique and most significant box for an object
          car_count.append(len(indexes))
          elapsed_time_1 = time.time() - starting_time # Calculate elapsed time to process a frame
          fps_1=frame_id/elapsed_time_1 # Calculating FPS or throughput for Query 1
          fps_list1.append(fps_1)
          time_1.append(elapsed_time_1)
          try:
            if query_choice =='Q2': #If the query choice is Q2 then processeach object for attribute or car class type classification
              for i in range(len(boxes)): # To process each object detected
                if i in indexes: # If the box is unique for an object
                  x, y, w, h = boxes[i] # Extract the position of the object box
                  roi = frame[y:y + h, x:x + w] # Extract trhe region of interest or cropping the car from the frame
                  if roi.shape[1] != 0:
                      img = cv2.resize(roi, (224, 224))/255 #Resize the car cropped image and normalize it to pass to classifier model
                      class_1 = model_classifier.predict(np.array([img])) # Pass the image to the classification model for car class prediction
                      class_arg = np.argmax(class_1) # Get the probabilties of class 0 and 1 and get the class with highest probability
                      label = str(classes[class_ids[i]]) # To get the label or type of object
                      if class_typ[class_arg] == 'sedan':
                          sedan_count += 1
                      elif class_typ[class_arg] == 'SUV':
                          suv_count += 1
                      else:
                          print("Error: Invalid car class")
                      confidence = confidences[i]
                      color = colors[class_ids[i]]
                      cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) # To create a box or rectangle on each object detected
                      cv2.putText(frame, label + " " + str(round(confidence, 2)) + ' ' + class_typ[class_arg],
                                  (x, y + 30), font, 1, (255, 255, 255), 2) # To put text like FPS, counts, car class on  box or rectangle on each object detected
            else: # If the query is Q1 then only perform object detection and not classification
              for i in range(len(boxes)):
                if i in indexes:
                  x,y,w,h = boxes[i]
                  label = str(classes[class_ids[i]])
                  confidence= confidences[i]
                  color = colors[class_ids[i]]
                  cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                  cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)
          except Exception as e:
              pass
          #To put information like FPS, Total count of cars, sedan and suv cars on the frame
          cv2.putText(frame,"Throughput(FPS) Q1: "+str(round(fps_1,2)),(10,10),font,1,(0,0,0),1)
          cv2.putText(frame,"Total Cars: "+str(len(indexes)),(10,40),font,1,(0,0,0),1)
          if query_choice == 'Q2':
            elapsed_time_2 = time.time() - starting_time
            fps_2=frame_id/elapsed_time_2
            fps_list2.append(fps_2)
            time_2.append(elapsed_time_2)
            sedan_list.append(sedan_count)  
            suv_list.append(suv_count) 
            cv2.putText(frame,"Throughput(FPS) Q2: "+str(round(fps_2,2)),(10,25),font,1,(0,0,0),1)
            cv2.putText(frame,"Total Sedan Cars: "+str(sedan_count),(10,55),font,1,(0,0,0),1)
            cv2.putText(frame,"Total SUV Cars: "+str(suv_count),(10,70),font,1,(0,0,0),1)
          
          writer.write(cv2.resize(frame,(428, 576)))
          pyplot.imshow(frame)
          pyplot.show()
          key = cv2.waitKey(1)  # wait 1ms the loop will start again and we will process the next frame

          if key == 27:  # esc key stops the process
              break;
          print("Consumed", frame_id)


        def fn(c): ## Calculating False negative for classification model
          if c['SUV_Count_Difference'] > 0:
              return (c['SUV_Count_Difference'] - abs(c['Car_Count_Difference']))
          else:
              return 0
        def fp(c): ## Calculating False positive for classification model
          if c['Sedan_Count_Difference'] > 0:
              return (c['Sedan_Count_Difference'] - abs(c['Car_Count_Difference']))
          else:
              return 0
    
        def f1_score_Q1(c):
          if c['Car_Count_Difference'] > 0:
            ratio = c['Car_Count_Predicted']/c['Total']
            return (2*ratio)/(1+ratio)
          else:
            return 1
        
        # Create dataframe which has frame id the total cars detected in each frame and if query is Q2 the nalso add sedan and suv car counts for each frame and then
        # calculate the F1 score for each query by comparing the predicted values with the actual groundtruth values
        df['Frame_Id'] =  frame_id_lst
        df['Car_Count_Predicted'] =  car_count
        true_df = pd.read_excel('Groundtruth.xlsx')
        df['Total'] = true_df['Total']
        df['Car_Count_Difference'] =  df['Total'] - df['Car_Count_Predicted']
        df['F1_Score_Q1'] = df.apply(f1_score_Q1, axis=1)
        print("Q1 F1 score is: {}".format(str(df['F1_Score_Q1'].mean())))

        if query_choice =='Q2':
          df['Sedan_Count_Predicted'] =  sedan_list
          df['SUV_Count_Predicted'] =  suv_list
          df['Sedan'] = true_df['Sedan']
          df['SUV'] = true_df['SUV']
          df['Sedan_Count_Difference'] =  df['Sedan'] - df['Sedan_Count_Predicted'] 
          df['SUV_Count_Difference'] =  df['SUV'] - df['SUV_Count_Predicted']
          df['true_positives'] = df[['Sedan','Sedan_Count_Predicted']].min(axis=1) + df[['SUV','SUV_Count_Predicted']].min(axis=1)
          df['false_positives'] = df.apply(fp, axis=1)
          df['false_negatives'] = df.apply(fn, axis=1)
          df['Precision'] = df['true_positives']/(df['true_positives']+df['false_positives'])
          df['Recall'] = df['true_positives']/(df['true_positives']+df['false_negatives'])
          df['F1_Score_Q2'] = (2*df['Precision']*df['Recall'])/(df['Precision']+df['Recall'])
          print("Q2 F1 score is: {}".format(str(df['F1_Score_Q2'].mean())))

        df.to_csv('Predicted_Output.csv')
        print("Successful")

        cap.release()
        cv2.destroyAllWindows()
        end = time.time()
        print(end - starting_time)
        
starting_time= time.time()

#Run producer and consumer thread
ProducerThread().start()
ConsumerThread().start()


# In[ ]:


import matplotlib.pyplot as plt
pyplot.figure()
pyplot.plot(time_1, fps_list1)
pyplot.xlabel('Time (s)')
pyplot.ylabel('FPS - Q1')
pyplot.show()


# In[ ]:


plt.figure()
plt.plot(time_2, fps_list2)
plt.xlabel('Time (s)')
plt.ylabel('FPS - Q2')
plt.show()


# In[ ]:




