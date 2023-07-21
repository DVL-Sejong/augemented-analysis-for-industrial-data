import os
import csv

datadir = "./fix/"
datalist = os.listdir(datadir)

for dataName in datalist:
    _path = datadir + dataName +"/"
    classList = os.listdir(_path)
    for className in classList:
        _path = datadir + dataName +"/"+ className +"/"
        stiDirNameList = os.listdir(_path)
        for stiDirName in stiDirNameList:
            _path = datadir + dataName +"/"+ className +"/"+ stiDirName +"/"
            fileList = os.listdir(_path)
            for fileName in fileList:
                _path = datadir + dataName +"/"+ className +"/"+ stiDirName +"/"+ fileName
                if os.path.getsize(_path) == 0:
                    print(_path)
                    # os.remove(_path)