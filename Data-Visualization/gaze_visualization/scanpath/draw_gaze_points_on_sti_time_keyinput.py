import os
import numpy as np
import pandas as pd
import cv2

STI_DIR = "./all_data_sti"
# STI_DIR = "./sti_Ehinger"
DATA_DIR = "./eye tracking data (collected)"
IMG_EXE = ["jpg", "jpeg", "png"]

_rf = "c0+social_019__user09.csv"
OUT_DIR = "./draw_points_time/"+_rf.split('.')[0]+"/"

gaze_path = DATA_DIR+"/"+_rf
fileName = _rf.split('.')[0].split('__')[0]
userName = _rf.split('.')[0].split('__')[1]

stiPath = STI_DIR+"/"+fileName+"."+IMG_EXE[0]
if not(os.path.isfile(stiPath)):
        stiPath = STI_DIR+"/"+fileName+"."+IMG_EXE[1]
        if not(os.path.isfile(stiPath)):
            stiPath = STI_DIR+"/"+fileName+"."+IMG_EXE[2]

df = pd.read_csv(gaze_path)
gaze_len = len(df.values.tolist())
count_val = 1
ALL_SAVE_FLAG = False
IMAGE = cv2.imread(stiPath)
while True:
    cv2.imshow("image", IMAGE)
    # draw points until count
    for index, row in df.iterrows():
        if index == count_val:
            break
        x = row['x']
        y = row['y']
        c = (255, 255, 255)
        if index == count_val-1:
            c = (0, 0, 255)
        cv2.circle(IMAGE, (int(x), int(y)), 6, (0,0,0), -1)
        cv2.circle(IMAGE, (int(x), int(y)), 5, c, -1)
    if ALL_SAVE_FLAG == True:
        if count_val > len(df):
            break
        out_path = OUT_DIR+"/"+fileName+"-"+str(count_val-1).zfill(3)+".png"
        cv2.imwrite(out_path, IMAGE)
        print("ALL_SAVE_FLAG: "+out_path)
        count_val = count_val+1
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('n'):
        IMAGE = cv2.imread(stiPath)
        if count_val == len(df):
            count_val == len(df)
        else:
            count_val = count_val+1
        print(count_val-1)
        continue
    elif key == ord('b'):
        IMAGE = cv2.imread(stiPath)
        if count_val <= 1:
            count_val = 1
        else:
            count_val = count_val-1
        # print(count_val)
        continue
    elif key == ord('s'):
        out_path = OUT_DIR+"/"+fileName+"-"+str(count_val-1).zfill(3)+".png"
        cv2.imwrite(out_path, IMAGE)
        print(out_path)
        continue
    elif key == ord('A'):
        if ALL_SAVE_FLAG == False:
            ALL_SAVE_FLAG = True
        else:
            ALL_SAVE_FLAG = False
        continue