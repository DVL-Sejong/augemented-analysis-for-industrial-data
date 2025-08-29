import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


RAW_DIR = "./eye tracking data (Ehinger)"
STI_DIR = "./sti_Ehinger"
# RAW_DIR = "./eye tracking data (collected)"
# STI_DIR = "./all_data_sti"
OUT_DIR = "./all_draw_scanpath/Ehinger"
# OUT_DIR = "./scanpath_raw_label_without_line"
# OUT_DIR = "./scanpath_raw_label_only_fix"

IMG_EXE = ["jpg", "jpeg", "png"]

csv_list = os.listdir(RAW_DIR)

for _rf in csv_list:
    print(_rf)
    fileName = _rf.split('.')[0].split('_')[0]+"_"+_rf.split('.')[0].split('_')[1]
    userName = _rf.split('.')[0].split('_')[2]
    csvPath = RAW_DIR+"/"+_rf
    stiPath = STI_DIR+"/"+fileName+"."+IMG_EXE[0]

    if not(os.path.isfile(stiPath)):
        stiPath = STI_DIR+"/"+fileName+"."+IMG_EXE[1]
        if not(os.path.isfile(stiPath)):
            stiPath = STI_DIR+"/"+fileName+"."+IMG_EXE[2]
            if not(os.path.isfile(stiPath)):
                print("No stimulus")
                continue

    img = Image.open(stiPath)
    img_draw = ImageDraw.Draw(img)

    df = pd.read_csv(csvPath)
    df = df.dropna()
    # draw lines
    prevCoordi = [-999, -999]
    for index, row in df.iterrows():
        _x = int(row[0])
        _y = int(row[1])
        if index == 0:
            prevCoordi = [_x, _y]
            continue
        img_draw.line((prevCoordi[0], prevCoordi[1], _x, _y), fill="black", width=4)
        img_draw.line((prevCoordi[0], prevCoordi[1], _x, _y), fill="white", width=2)
        
        prevCoordi = [_x, _y]

    # draw points
    # for index, row in df.iterrows():
    #     _x = int(row[0])
    #     _y = int(row[1])
    #     _l = int(row[2])
    #     if index == 0:
    #         img_draw.ellipse([(_x-4, _y-4), (_x+4, _y+4)], fill="red", outline="black")
    #     else:
    #         if _l == 0:
    #             img_draw.ellipse([(_x-2, _y-2), (_x+2, _y+2)], fill="blue", outline="green")
    #         elif _l == 1:
    #             img_draw.ellipse([(_x-2, _y-2), (_x+2, _y+2)], fill="black", outline="green")
    #         else:
    #             img_draw.ellipse([(_x-2, _y-2), (_x+2, _y+2)], fill="white", outline="green")
    for index, row in df.iterrows():
        _x = int(row[0])
        _y = int(row[1])
        _l = int(row[2])
        
        # only fix
        # if _l == 0:
        #     img_draw.ellipse([(_x-4, _y-4), (_x+4, _y+4)], fill="white", outline=(115, 115, 115))
        # else:
        #     img_draw.ellipse([(_x-4, _y-4), (_x+4, _y+4)], fill="black", outline=(115, 115, 115))
        
        # all events
        img_draw.ellipse([(_x-4, _y-4), (_x+4, _y+4)], fill="white", outline=(115, 115, 115))

            

    outPath = OUT_DIR+"/"+fileName+"_"+userName+".png"
    img.save(outPath)