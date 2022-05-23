import os
from ml_model import *
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def hi():
    return jsonify({"message" : "Hey there!"})

@app.route('/predict', methods = ['POST'])
def predict():
    phrase = request.json['message']
    res = pre(phrase)
    return jsonify({"sentiment" : res})

if __name__ == '__main__':
    port = 8000
    app.run(debug=True, port=port)