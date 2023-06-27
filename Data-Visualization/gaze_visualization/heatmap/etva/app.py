# -*- coding: utf-8 -*-
from flask import Flask, abort,render_template, redirect, request, url_for, json, send_file, Response,jsonify, Blueprint, g
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import platform

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

#시스템 판별 Windows Linux Darwin
OS_system = platform.system()

#특정 path위치의 1단계 하위리스트 출력 -> 실험자 목록이나 이미지 목록 리스트 출력.
def load_dir_list(direc):
    direcs = os.listdir(direc)
    resdic = []
    for di in direcs:
        #full_dir = os.path.join(direc, di)
        resdic.append(di.split('.')[0])
    return resdic

#print(OS_system)
#if OS_system == "Windows":
#    raw_data = load_dir_list(os.getcwd()+'\\static\\data\\eye_features')
#elif OS_system == "Darwin":
#    raw_data = load_dir_list(os.getcwd()+'/static/data/eye_features')

#rawEyeData = {}
#df = pd.read_csv('static/data/eye_features/'+"U0121_1RTE_0"+'.csv',sep=",", dtype='unicode',header=0)
#rawEyeData["gaze"] = {'x':df['x'].tolist(), 'y':df['y'].tolist()}

#stiData = {}
#sdf = pd.read_csv('static/data/eye_features/'+"sti"+'.csv',sep=",", dtype='unicode',header=0)
#stiData = {'id':sdf['id'].tolist(), 'center':sdf['center'].tolist(), 'conc':sdf['conc'].tolist(), 'cono':sdf['cono'].tolist(), 'curr':sdf['curr'].tolist()}

#@app.route("/")
#def home():
#    return render_template('home.html', rawEyeData=rawEyeData, stiData=stiData)

@app.route("/")
def home():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)