import cv2
import numpy as np
import os

if __name__ == '__main__':
    # 1. scan clips
    clip = './clips/test_clip.h264'
    # 2. preview and capture images
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )
    print("FPS: " + str(fps))
    ret,frame = cap.read()
    print("Image size: " + str(frame.shape))
    # 2.2 preview clip         
    while True:
        # 2.2.1 read image            
        ret,raw_img = cap.read()     
        if not ret:
            break
        cv2.imshow('Preview', raw_img)        
        key = cv2.waitKey(int(1000/fps))
        # 2.2.3 process commands
        if key == ord('q'): # Press q for quit
            break
    cap.release()
