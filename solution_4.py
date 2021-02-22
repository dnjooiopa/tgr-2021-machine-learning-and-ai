import cv2
import numpy as np
import tensorflow as tf
import time
from step_2_1 import detect
from object_detection.utils import visualization_utils as viz_utils

# globals
lime_count = 0
marker_count = 0

def detect_objects(img):
    input_tensor = tf.convert_to_tensor([img], dtype=tf.float32)
    results = detect(input_tensor)
    bboxes = results['detection_boxes'][0].numpy()
    classes = results['detection_classes'][0].numpy().astype(np.uint32) + 1
    scores = results['detection_scores'][0].numpy()

    return classes[scores > 0.8], bboxes[scores > 0.8], scores[scores > 0.8]

category_index = {1: {'id': 1, 'name': 'lime'}, 2: {'id': 2, 'name': 'marker'},}
dummy_scores = np.array([1], dtype=np.float32)

def overlay_objects(image_np, bboxes, classes, scores):
    image_with_annotation = cv2.cvtColor(image_np.copy(), cv2.COLOR_BGR2RGB)
    viz_utils.visualize_boxes_and_labels_on_image_array(
      image_with_annotation,
      bboxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=0.8)
    
    return image_with_annotation.copy()


if __name__ == "__main__":
    img = cv2.imread('./data/test/lime031.jpg')
    classes, bboxes, scores = detect_objects(img)
    #image_overlay = overlay_objects(img, bboxes, classes, scores)
    # cv2.imshow('hello', image_overlay)
    # cv2.waitKey(0)
    # print(results)

    cap = cv2.VideoCapture('data/clips/test_clip.h264')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )
    #sqsize = max(width,height)

    new_width = 320
    new_height = int(height * (new_width / width))
    sqsize = 320
    count  = 0
    print(width, height)
    while True:
        # capture image
        ret, raw_img = cap.read()
        raw_img = cv2.resize(raw_img, (new_width, new_height))
        if not ret:
            break
        # add margin
        frame = np.zeros((sqsize,sqsize,3), np.uint8)
        if new_width > new_height:
            offset = int( (new_width - new_height)/2 )
            frame[offset:new_height+offset,:] = raw_img
        else:
            offset = int( (new_height - new_width)/2 )
            frame[:,offset:] = raw_img
        img = frame
        if count == 5:
            start_time = time.time()
            classes, bboxes, scores = detect_objects(frame)
            end_time = time.time()
            elapsed_time = end_time - start_time
            #print('elapsed_time (tf):', elapsed_time)
            img = overlay_objects(frame, bboxes, classes, scores)
            count = 0

            # overlay with line
            margin = 50
            pt1 = ( int(sqsize/2-margin), 0 )
            pt2 = ( int(sqsize/2-margin), int(sqsize) )
            cv2.line(img, pt1, pt2, (0,0,255), 2)
            pt1 = ( int(sqsize/2+margin), 0 )
            pt2 = ( int(sqsize/2+margin), int(sqsize) )
            cv2.line(img, pt1, pt2, (0,0,255), 2)        

            # count if obj have passed the line
            for box in bboxes:
                
                if int(box[1] * 320) > int(sqsize/2-margin) and int(box[1] * 320) < int(sqsize/2+margin) and int(box[3] * 320) > int(sqsize/2-margin) and int(box[3] * 320) < int(sqsize/2+margin) :
                    if classes[0] == 1:
                        lime_count = lime_count + 1
                    if classes[0] == 2:
                        marker_count = marker_count + 1
                    print('Lime:', lime_count)
                    print('Marker:', marker_count)

            # preview image
            cv2.imshow('Preview', img)
            count = 0

        # delay_time = int(1000/fps)     
        count = count + 1
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()
    #print_result()


