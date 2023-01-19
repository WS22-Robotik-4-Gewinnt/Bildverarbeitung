import os


def takePicture():
    os.system('fswebcam --no-banner -S 30 - r 480x320 ./images/image1.jpg')
    return ("fine")
