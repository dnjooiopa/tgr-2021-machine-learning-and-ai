import cv2
import numpy as np
import os

if __name__ == '__main__':
    # 1 scan clips
    CLIP_PATH = './data/clips'
    flist = os.listdir(CLIP_PATH)
    clips = []
    for f in flist:
        if f.endwith('.h264'):
            clips.append(CLIP_PATH + f)
    print(clips)
    # 2 preview and capture images
    idx = 0
    running = True
    IMG_PATH = './data/clips'
    for clip in clips:
        # 2.1 import clip
        print("Reading", clip)
        cap = cv2.VideoCapture(clip)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sqsize = max(width, height)
        print("FPS:", str(fps))
        ret, frame = cap.read()
        print("Image size:", str(frame.shape))
        # 2.2 preview clip
        while True:
            # 2.2.1 read image
            ret, raw_img = cap.read()
            if not ret:
                break
            frame = np.zeros((sqsize, sqsize, 3), np.uint8)
            if key == ord('n'): 
