import os
import numpy as np
import pandas as pd
import cv2

DATA_FILE_PATH = "./2-5-0.csv"
STI_DIR = "./sti_target_box/"
STI_PATH = STI_DIR+DATA_FILE_PATH.split('/')[1].split('.')[0]+".png"

df = pd.read_csv(DATA_FILE_PATH)

gaze_len = len(df.values.tolist())
count_val = 1
IMAGE = cv2.imread(STI_PATH)
while True:
    cv2.imshow("Len(gaze) = "+str(gaze_len), IMAGE)
    # draw points until count
    for index, row in df.iterrows():
        if index == count_val:
            break
        x = int(row['x'])
        y = int(row['y'])
        target = int(row['target'])
        c = (255, 255, 255)
        if target == 0:
            c = (255, 0, 0)
        elif target == 1:
            c = (0, 255, 0)
        elif target == 2:
            c = (255, 255, 0)
        else:
            c = (0, 0, 0)
        cv2.circle(IMAGE, (int(x), int(y)), 6, (0,0,255), -1)
        cv2.circle(IMAGE, (int(x), int(y)), 5, c, -1)
    
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('n'):
        IMAGE = cv2.imread(STI_PATH)
        if count_val == len(df):
            count_val == len(df)
        else:
            count_val = count_val+1
        print(count_val-1)
        continue
    elif key == ord('b'):
        IMAGE = cv2.imread(STI_PATH)
        if count_val <= 1:
            count_val = 1
        else:
            count_val = count_val-1
        # print(count_val)
        continue