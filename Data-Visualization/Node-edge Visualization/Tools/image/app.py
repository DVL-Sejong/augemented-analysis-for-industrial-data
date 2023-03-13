import os
import math

import numpy as np
import pandas as pd

import cv2
import seaborn as sns

STI_DIR = "./stimuli/"
STI_LIST = os.listdir(STI_DIR)
OUT_DIR = "./segmentation/"

CLICKS = []
AREAS = []
IMAGE = 0
mouse_x, mouse_y = -1, -1
mouse_down_flag = False
mouse_move_x, mouse_move_y = -1, -1
STIMULUS_INDEX = 0

PALETTE = sns.color_palette(None, 50)
palette_index = 0
for _color_float in PALETTE:
  _c_int_255 = (int(_color_float[0]*255), int(_color_float[1]*255), int(_color_float[2]*255))
  PALETTE[palette_index] = _c_int_255
  palette_index = palette_index+1

def mouse_callback(event, x, y, flags, param):
  global CLICKS, mouse_x, mouse_y, mouse_down_flag, mouse_move_x, mouse_move_y
  if event == cv2.EVENT_LBUTTONUP:
    mouse_x, mouse_y = x, y
    # print("("+str(mouse_x)+", "+str(mouse_y)+")")
    CLICKS.append([mouse_x, mouse_y])
    # print(CLICKS)
    if mouse_down_flag == False:
      mouse_down_flag = True
  elif event == cv2.EVENT_MOUSEMOVE:
    mouse_move_x, mouse_move_y = x, y

  draw_circle()

def draw_circle():
  global IMAGE
  if len(AREAS) != 0:
    colorIdx = 0
    for area in AREAS:
      count = 0
      prevPt = AREAS[0]
      for point in area:
        if count != 0:
          cv2.line(IMAGE, (prevPt[0], prevPt[1]), (point[0], point[1]), PALETTE[colorIdx], 2)
        cv2.circle(IMAGE, (point[0], point[1]), 4, PALETTE[colorIdx], -1)
        count = count+1
        prevPt = point
      # print(np.array(area))
      # cv2.drawContours(IMAGE, np.array(area), contourIdx=-1, color=(0,0,255), thickness=-1)
      cv2.fillPoly(IMAGE, pts=[np.array(area)], color=PALETTE[colorIdx])
      colorIdx = colorIdx+1

  if len(CLICKS) != 0 and len(CLICKS) < 2:
    # lastPt = []
    for point in CLICKS:
      cv2.circle(IMAGE, (point[0], point[1]), 4, (0,0,255), -1)
      # lastPt = point
    # cv2.line(IMAGE, (lastPt[0], lastPt[1]), (mouse_move_x, mouse_move_y), (0,0,255), 2)
  elif len(CLICKS) >= 2:
    count = 0
    prevPt = CLICKS[0]
    # lastPt = []
    for point in CLICKS:
      cv2.circle(IMAGE, (point[0], point[1]), 4, (0,0,255), -1)
      if count != 0:
        cv2.line(IMAGE, (prevPt[0], prevPt[1]), (point[0], point[1]), (0,0,255), 2)
      count = count+1
      prevPt = point
      # lastPt = point
    # cv2.line(IMAGE, (lastPt[0], lastPt[1]), (mouse_move_x, mouse_move_y), (0,0,255), 2)




sti_path = STI_DIR+STI_LIST[STIMULUS_INDEX]
IMAGE = cv2.imread(sti_path)
while True:
  cv2.imshow("iamge", IMAGE)
  cv2.setMouseCallback("iamge", mouse_callback, IMAGE)
  key = cv2.waitKey(1)
  if key == 27:
    break
  elif key == ord('a'):
    IMAGE = cv2.imread(sti_path)
    if len(CLICKS) < 3:
      continue
    area = []
    fPt = CLICKS[0]
    for pt in CLICKS:
      area.append([pt[0], pt[1]])
    area.append([fPt[0], fPt[1]])
    AREAS.append(area)

    CLICKS = []
    if mouse_down_flag == True:
      mouse_down_flag = False
    continue
  elif key == ord('n'):
    if len(STI_LIST) > STIMULUS_INDEX+1:
      # segmentation data save process
      img_height, img_width = IMAGE.shape[:2]
      areaCount = 1
      allPixels = np.zeros(IMAGE.shape[:2])
      for area in AREAS:
        for _r in range(0, img_height):
          for _c in range(0, img_width):
            if allPixels[_r][_c] == 0:
              innerCheck = cv2.pointPolygonTest(np.array(area), (_c, _r), False)
              if innerCheck >= 0:
                allPixels[_r][_c] = areaCount
                # print("("+str(_c)+", "+str(_r)+")")
        areaCount = areaCount+1
      allPixels_df = pd.DataFrame(allPixels, columns=None)
      out_path = OUT_DIR+STI_LIST[STIMULUS_INDEX].split('.')[0]+".csv"
      allPixels_df.to_csv(out_path, index=False, header=False)
      print(out_path)
      AREAS = []
      CLICKS = []

      STIMULUS_INDEX = STIMULUS_INDEX+1
      sti_path = STI_DIR+STI_LIST[STIMULUS_INDEX]
      IMAGE = cv2.imread(sti_path)
    continue
  elif key == ord('b'):
    if STIMULUS_INDEX > 0:
      STIMULUS_INDEX = STIMULUS_INDEX-1
      sti_path = STI_DIR+STI_LIST[STIMULUS_INDEX]
      IMAGE = cv2.imread(sti_path)
    continue
  elif key == ord('r'):
    IMAGE = cv2.imread(sti_path)
    if len(CLICKS) >= 1:
      CLICKS.pop()
    continue
  elif key == ord('-'):
    IMAGE = cv2.imread(sti_path)
    if len(AREAS) >= 1:
      AREAS.pop()
    continue

cv2.destroyAllWindows()