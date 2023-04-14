import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

CSV_DRI = "./all_data_segmentation_csvs"
STI_DIR = "./all_data_sti"
OUT_DIR = "./all_draw_gaze_points"

IMG_EXE = ["jpg", "jpeg", "png"]

csv_list = os.listdir(CSV_DRI)

# for _rf in csv_list:
_rf = "social_043.csv"
fileName = _rf.split('.')[0].split('__')[0]
# userName = _rf.split('.')[0].split('__')[1]
userName = "user001"
csvPath = CSV_DRI+"/"+_rf
# stiPath = STI_DIR+"/c0+"+fileName+"."+IMG_EXE[0]
stiPath = STI_DIR+"/c0+social_043.jpg"

# if not(os.path.isfile(stiPath)):
#     stiPath = STI_DIR+"/"+fileName+"."+IMG_EXE[1]
#     if not(os.path.isfile(stiPath)):
#         stiPath = STI_DIR+"/"+fileName+"."+IMG_EXE[2]

img = Image.open(stiPath)
img_draw = ImageDraw.Draw(img)

df = pd.read_csv(csvPath)
for index, row in df.iterrows():
    _x = int(row['x'])
    _y = int(row['y'])
    if index == 0:
        img_draw.ellipse([(_x-5, _y-5), (_x+5, _y+5)], outline="red")
        #img_draw.ellipse([(_x-5, _y-5), (_x+5, _y+5)], fill="blue", outline="red")
    else:
        img_draw.ellipse([(_x-5, _y-5), (_x+5, _y+5)], outline="black")

outPath = OUT_DIR+"/"+fileName+"__"+userName+".png"
img.save(outPath)