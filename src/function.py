import logging
import os
import cv2
from grid import bilderkennung


def takePicture():
    cam = cv2.VideoCapture(0)
    ret, image = cam.read()
    cv2.imwrite('./testimage.jpg', image)
    cam.release()
    bilderkennung('./image_demo.jpg')
    return ("fine")


#takePicture() #TODO für Marcel zum testen ansonsten weg
