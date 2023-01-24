import logging
import os
import cv2
from grid import _main


def takePicture():
    # os.system('fswebcam --no-banner -S 30 - r 480x320 /images/image1.jpg')
    test = 4
    logging.info(str(test))
    cam = cv2.VideoCapture(0)

    while True:
        ret, image = cam.read()
        #cv2.imshow('Imagetest',image)
        #k = cv2.waitKey(1)
        #if k != -1:
        break
    cv2.imwrite('./testimage.jpg', image)
    cam.release()
    #cv2.destroyAllWindows()
    _main('./image_demo.jpg')
    return ("fine")

takePicture()


