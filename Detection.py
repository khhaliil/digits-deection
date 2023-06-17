import tensorflow as tf
import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt

import tensorflow as tf


cameraNum = 0
path = r"modelTrained1\modelTrained2.p"
modelPath = r"C:\Users\MSI\PycharmProjects\pythonProject\Textdetection\model"
limite = 0.8
############################


# Load the entire saved model

cap = cv2.VideoCapture(cameraNum)
pickle_in = open(path,"rb")
model = pickle.load(pickle_in)
cap.set(15, 15 )
cap.set(3, 640)
cap.set(4, 480)


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(
                    imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(
                    imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


while True:
    _, imgorg = cap.read()
   # imgorg= cv2.flip(imgorg,0)
    imgorg=cv2.GaussianBlur(imgorg,(5,5),0)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(imgorg, -1, kernel)

    #imgorg=cv2.flip(imgorg,1)
    imgflip = cv2.flip(imgorg,1)
    img = np.asarray(sharpened)
    img = cv2.resize(img, (50, 50))
    img1 = preprocess(img)
    imgp = img1.reshape(1, 50, 50, 1)

    prediction = model.predict(imgp)
    probval = np.amax(prediction)
    classIndexes=np.argmax(prediction,axis=1)
    if probval>limite:
        print(" this is classs indexe   ", classIndexes[0], "   de prob egal à ", probval * 100)
        cv2.putText(sharpened, '[' +str(classIndexes[0])+']'+ ' '+str(round(probval*100,2))
                    ,(75,75),
                    cv2.FONT_HERSHEY_COMPLEX,
                    3,(190,255,0),3)

    #print(predictions, " ", classindex)
    cv2.imshow("proc", sharpened)
    cv2.imshow("procecced",cv2.resize(img1,(320,320)))

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

