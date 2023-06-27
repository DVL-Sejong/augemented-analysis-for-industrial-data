import sys
import os

import numpy as np
import pandas as pd
import json
import cv2
from flask import *
from flask_cors import CORS

DATASET = ""

app = Flask(__name__)
if __name__ == '__main__':
  app.jinja_env.auto_reload = True
  app.config['TEMPLATES_AUTO_RELOAD'] = True
  app.run(debug=True)
CORS(app)


@app.route('api/loadData', methods=['POST'])
def loadData():
  global DATASET
  print(request.form)
  response = {}
  try:
    DATASET = request.form['DATASET']
  

    response['status'] = 'success'
  except Exception as e:
    response['status'] = 'failed'
    response['reason'] = e
    print(e)
  return json.dumps(response)