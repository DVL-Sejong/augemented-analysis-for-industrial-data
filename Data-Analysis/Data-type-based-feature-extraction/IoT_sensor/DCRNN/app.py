from flask import *

import pandas as pd
import json
import requests
import time

#If you run DCRNN, please remove the comment below
'''
from static.lib.DCRNN.lib.utils import load_graph_data
from static.lib.DCRNN.model.dcrnn_supervisor import DCRNNSupervisor

import tensorflow as tf
import yaml
'''

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video/<channel>')
def video(channel):
    source = ''
    url = 'https://map.naver.com/v5/api/cctv/list?channel=' + channel
    data = requests.get(url).json()

    for cctv in data['message']['result']['cctvList']:
        if cctv['channel'] == int(channel):
            liveParam = cctv['liveEncryptedString']
            if liveParam is None:
                liveParam = cctv['encryptedString']

            url = 'http://cctvsec.ktict.co.kr/' + channel + '/' + liveParam
            print('http://cctvsec.ktict.co.kr/' + channel + '/' + liveParam)

            headers = {
                'Accept': '*/*',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
                'Connection': 'keep-alive',
                'Host': 'cctvsec.ktict.co.kr',
                'Set-Fetch-Dest': 'empty',
                'Set-Fetch-Mode': 'cors',
                'Set-Fetch-Site': 'cross-site',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            source = response.url

            print(source)
            break

    return render_template('video.html', source=source)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    print('predict')
    time.sleep(10)

    #run_dcrnn() #If you run DCRNN, please remove the comment

    return 'Predict'

@app.route('/fileupload', methods=['GET', 'POST'])
def file_upload():
    if 'files[]' not in request.files:
        return 'Error'

    files = request.files.getlist('files[]')
    path = 'static/data/'
    for file in files:
        filename = file.filename
        file.save(path + filename)

    return 'success'

@app.route('/visfile', methods=['GET', 'POST'])
def vis_save():
    data = json.loads(request.get_data())
    df = pd.DataFrame.from_dict(data['data'])
    df.set_index(keys=['key'], inplace=True)
    df = df.astype('int')

    df.to_csv('static/data/' + data['type'] + '.csv', encoding='CP949')

    return 'Done'

@app.route('/export', methods=['GET', 'POST'])
def export_file():
    data = json.loads(request.get_data())

    with open('static/data/predict_data.json', 'w', encoding='UTF-8-sig') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return 'Good'

#If you run DCRNN, please remove the comment below
'''
def run_dcrnn():
    config_filename = 'static/lib/DCRNN/data/model/Gangnam_trained/config_100.yaml'

    with open(config_filename) as f:
        config = yaml.safe_load(f)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    graph_pkl_filename = config['data']['graph_pkl_filename']
    _, _, adj_mx = load_graph_data(graph_pkl_filename)
    print(adj_mx.shape)

    with tf.Session(config=tf_config) as sess:
        supervisor = DCRNNSupervisor(adj_mx = adj_mx, **config)
        supervisor.load(sess, config['train']['model_filename'])
        outputs = supervisor.evaluate(sess)

        for i in range(12):
            origin = pd.DataFrame(
                outputs['groundtruth'][i].reshape(len(outputs['groundtruth'][0]),
                                                  len(outputs['groundtruth'][0][0]))
            )
            predict = pd.DataFrame(
                outputs['predictions'][i].reshape(len(outputs['predictions'][0]),
                                                  len(outputs['predictions'][0][0]))
            )
            origin.to_csv(f'static/data/DCRNN/prediction/after_{i+1}h_origin.csv')
            predict.to_csv(f'static/data/DCRNN/prediction/after_{i+1}h_predict.csv')
        print('Save')
'''