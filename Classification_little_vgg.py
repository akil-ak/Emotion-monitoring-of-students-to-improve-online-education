from keras.models import load_model
import streamlit as st
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import os
import time
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import psutil
import matplotlib.pyplot as plt
import random
from itertools import count
from matplotlib.animation import FuncAnimation




face_classifier = cv2.CascadeClassifier(r'C:\Users\User\OneDrive\Documents\semester-3\design thinking\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\User\OneDrive\Documents\semester-3\design thinking\Emotion_little_vgg.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

cap = cv2.VideoCapture(0)
Angry=0
suprise=0
sad=0
happy=0
neutral=0
import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import random
import time

while True:
    # Grab a single frame of video
    ret, frame = cap.read()

    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    # rect,face,image = face_detector(frame)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            #print(preds)
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            
        

            now = datetime.now()

            current_time = (now.strftime("%H%M%S"))
            #print("Current Time =", current_time, " label: ",label)
            
            
            localtime = time.localtime()
            result = time.strftime("%I:%M:%S %p", localtime)
           # print(result)

            if(label=='Angry'):
                Angry=1
            if(label=='Neutral'):
               neutral=1
            if(label=='Suprise'):
                suprise=1
            if(label=='Happy'):
               happy=1
            if(label=='Sad'):
               sad=1
            f=open("text.txt",'a')
            f.write(label)
            f.write(" : " )
            f.write(current_time)
            f.write("\n")
            
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Emotion Detector',frame)
    


    
    '''data =[Angry,neutral,suprise,happy,sad] 
    mycolors = ["black", "hotpink", "b", "#4CAF50","Red"]
    myexplode = [0.2, 0.2, 0.2, 0.2,0.2]
 
# Creating plot
    fig = plt.figure(figsize =(10, 7))
    plt.pie(data, labels =class_labels ,colors=mycolors,wedgeprops = {"edgecolor" : "black",
                      'linewidth': 2,
                      'antialiased': True})
    plt.legend()

  # show plot
    plt.show()'''
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        plt.close()
        break
'''print(Angry)
print(neutral)
print(suprise)
print(happy)
print(sad)'''

        
'''
Year = []
Year.append(result)
Unemployment_Rate = [1,neutral]
print(Year)
print(Unemployment_Rate)
  
plt.plot(Year, Unemployment_Rate)
plt.title('Unemployment Rate Vs Year')
plt.xlabel('Year')
plt.ylabel('Unemployment Rate')
plt.show()'''
    
  
# Creating dataset



cap.release()
cv2.destroyAllWindows()


























