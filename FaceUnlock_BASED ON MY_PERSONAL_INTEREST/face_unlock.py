import cv2 as cv
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import random as random
import numpy as np


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D

def detect_face() :
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Camera cannot be opened")
        exit()
    
    i = 0
    while True:
        ret,frame = cap.read()
        if not ret:
            print("Can't receive frame")
            break
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        # haar_face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
        haar_face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

        faces = haar_face_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5);
        print("No of face: " + str(len(faces)))
        for(x,y,w,h) in faces:
            cv.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),2)

        if(i == 100):
            break

        file = 'Faces/image' + str(i) + '.png'

        if(len(faces) != 0):
            cv.imwrite(file,gray[y:y+h, x:x+w])
            i = i + 1

        cv.imshow('Camera',gray)

        if(cv.waitKey(1) == ord('q')):
            break
    cap.release()
    cv.destroyAllWindows()

def automate_script():
    dir2 = 'test/'
    i = 0
    for folder in os.listdir(dir2):
        try:
            new_path = dir2 + folder + '/'
            print(new_path)
            if(i == 100):
                    break    
            for img in os.listdir(new_path):
                try:
                    img_arr1 = cv.imread(os.path.join(new_path,img),0)
                    # cv.imshow("asdasd",img_arr1)
                    haar_face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
                    faces = haar_face_cascade.detectMultiScale(img_arr1,scaleFactor=1.1,minNeighbors=5);
                    for(x,y,w,h) in faces:
                        cv.rectangle(img_arr1,(x,y),(x+w,y+h),(0,255,0),2)

                    if(i == 100):
                        break
                    file = 'Faces2/image' + str(i) + '.png'
                    if(len(faces) != 0):
                        cv.imwrite(file,img_arr1[y:y+h, x:x+w])
                        i = i + 1
                        
                except:
                    pass
        except:
            pass

def create_training_data():

    training_set = []
    dir1 = 'Faces/'
    dir2 = 'Faces2/'

    for img in os.listdir(dir2):
        try:
            img_arr = cv.imread(os.path.join(dir2,img),0)
            img_resize = cv.resize(img_arr,(50,50))
            training_set.append([img_resize,0])
        except:
            pass

    for img in os.listdir(dir1):
        try:
            img_arr = cv.imread(os.path.join(dir1,img),0)
            img_resize = cv.resize(img_arr,(50,50))
            training_set.append([img_resize,1])
        except:
            pass
    
    return training_set

def learn_face():
    training_data = []
    training_data = create_training_data()

    random.shuffle(training_data)
    x = []
    y = []
    for features,label in training_data:
        x.append(features)
        y.append(label)   
    
    x = np.array(x).reshape(-1,50,50,1)
    x = x.astype('float32')
    
    x /= 255.0
    y = np.array(y)
    model = Sequential()
    model.add(Conv2D(32,(3,3),input_shape = (50,50,1),activation='relu',padding='valid'))
    
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(64,(3,3),activation='relu'))
    
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Flatten())
    
    model.add(Dense(100,activation='relu'))

    model.add(Dense(1,activation='sigmoid'))
    print(model.summary())
    model.compile(loss= 'binary_crossentropy',
                 optimizer = 'adam',
                 metrics = ['accuracy'])
    model.fit(x,y,batch_size = 32,epochs = 12, validation_split=0.1)
    return model

def test_face(model):
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Camera cannot be opened")
        exit()
    i = 1;
    while(i):
        ret,frame = cap.read()
        if not ret:
            print("Can't receive frame")
            break
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        haar_face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

        faces = haar_face_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5);
        for(x,y,w,h) in faces:
            cv.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),2)

        file = 'MyImage' + '.png'

        if(len(faces) != 0):
            cv.imwrite(file,gray[y:y+h, x:x+w])
            
            i = i - 1

        cv.imshow('Camera',gray)

        if(cv.waitKey(1) == ord('q')):
            break
    cap.release()
    cv.destroyAllWindows()

    img = cv.imread('MyImage.png',0)
    img_r = cv.resize(img,(50,50))
    cv.imshow('new',img_r)
    x = np.array(img_r).reshape(-1,50,50,1)

    prediction = model.predict([x])
    
    print(prediction[0][0])
    if(prediction[0][0] == 1):
        print("Face Matched")
    else:
        print("Matching not successfull..")
    
# Main funciton
detect_face()
model = learn_face()
test_face(model)