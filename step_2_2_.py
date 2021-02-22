import cv2
import numpy as np
import tensorflow as tf
import os
from step_2_2_problem import *


if __name__ == '__main__':
    cap = cv2.VideoCapture('data/clips/test_clip.h264')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )
    sqsize = max(width,height)
    while True:
        # capture image
        ret,raw_img = cap.read()
        if not ret:
            break
        # add margin
        frame = np.zeros((sqsize,sqsize,3), np.uint8)
        if width > height:
            offset = int( (width - height)/2 )
            frame[offset:height+offset,:] = raw_img
        else:
            offset = int( (height - width)/2 )
            frame[:,offset:] = raw_img
        # problems
        class_id, bbox = detect_objects(frame)
        img = overlay_objects(frame)
        # overlay with line
        pt1 = ( int(sqsize/2-100), 0 )
        pt2 = ( int(sqsize/2-100), int(sqsize) )
        cv2.line(img, pt1, pt2, (0,0,255), 2)
        pt1 = ( int(sqsize/2+100), 0 )
        pt2 = ( int(sqsize/2+100), int(sqsize) )
        cv2.line(img, pt1, pt2, (0,0,255), 2)        
        # preview image
        cv2.imshow('Preview', img)     
        key = cv2.waitKey(int(1000/fps))
        if key == ord('q'):
            break
    cap.release()
    print_result()