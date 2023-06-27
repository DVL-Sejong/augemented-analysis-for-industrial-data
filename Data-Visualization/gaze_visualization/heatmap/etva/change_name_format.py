import os

dirPath = "./indoor_outdoor_t/"
fileList = os.listdir(dirPath)

for fileName in fileList:
    imgName = fileName.split(".")[0]
    imgExt = fileName.split(".")[1]
    newName = imgName.split("_fixMap")[0]+"."+imgExt
    os.rename(dirPath+fileName, dirPath+newName)