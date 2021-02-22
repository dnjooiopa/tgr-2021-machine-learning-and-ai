import cv2
import numpy as np
import os

if __name__ == '__main__':
    # 1 scan clips
    CLIP_PATH = './data/clips/'
    flist = os.listdir(CLIP_PATH)
    clips = []
    for f in flist:
        if f.endswith('.h264'):
            clips.append(CLIP_PATH + f)
    print(clips)
    # 2 preview and capture images
    IMG_PATH = './data/images/'
    idx = len([x for x in os.listdir(IMG_PATH) if x.endswith(".jpg")]) 
    print(idx)
    running = True
   
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
            if width > height:
                offset = int((width - height)/2)
                frame[offset:height+offset,:] = raw_img
            else:
                offset = int((height - width)/2)
                frame[:,offset:] = raw_img
            # 2.2.2 preview image
            cv2.imshow('Preview', frame)
            key = cv2.waitKey(int(1000/fps))
            if key == ord('n'): 
                break
            if key == ord('q'):
                running = False
                break
            if key == 32:
                fname = IMG_PATH + 'lime' + f'{idx:03}' + '.jpg'
                print('Saving to ' + fname)
                cv2.imwrite(fname, frame)
                idx = idx + 1
        cap.release()
        if running is False:
            break
