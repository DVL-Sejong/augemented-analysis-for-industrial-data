from flask import *

app = Flask(__name__)

@app.route('/')
def hello_world():  # put application's code here
    print("python flask server run")
    return render_template('trajectory.html')

if __name__ == '__main__':
    app.run(debug=True)