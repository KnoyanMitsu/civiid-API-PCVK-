from flask import Blueprint, request, jsonify
from services.startup import clf, scaler
import numpy as np
import services.naive as naive

route = Blueprint('route', __name__)

@route.route('/')
def index():
    return jsonify({'message': 'OK'})

@route.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        return jsonify(naive.predict(request.files['file'])), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

