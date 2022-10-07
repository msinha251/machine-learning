from flask import Flask
from load_model import predict_single
from flask import request, jsonify
import json

app = Flask('credit-card')

dir_path = '.'

@app.route('/ping')
def ping():
    return 'PONG'

@app.route('/predict1', methods=['POST'])
def predict1():
    customer = request.get_json()
    print(customer)
    predict_score = predict_single(customer, 1)
    result = {'probability': predict_score}
    return jsonify(result)

@app.route('/predict2', methods=['POST'])
def predict2():
    customer = request.get_json()
    print(customer)
    predict_score = predict_single(customer, 2)
    result = {'probability': predict_score}
    return jsonify(result)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)