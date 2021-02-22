import cv2
import numpy as np
import tensorflow as tf


def detect_objects(img):
    class_id = 1
    bbox = [0,0,0,0]
    return (class_id, bbox)


def overlay_objects(img):
    return img.copy()

def print_result():
    print('Finished')