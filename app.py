"""
Flask Web App – Student Performance Prediction System
Run: python app.py
"""

from flask import Flask, render_template, request, jsonify
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from model import predict_student

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        family_map = {'low': 0, 'medium': 1, 'high': 2}
        gender_map = {'Male': 0, 'Female': 1}

        input_dict = {
            'study_hours_per_day':  float(data['study_hours']),
            'attendance_percentage': float(data['attendance']),
            'previous_score':        float(data['previous_score']),
            'sleep_hours':           float(data['sleep_hours']),
            'extra_activities':      int(data['extra_activities']),
            'internet_access':       int(data['internet_access']),
            'family_support':        family_map.get(data['family_support'], 1),
            'gender':                gender_map.get(data['gender'], 0),
        }

        result = predict_student(input_dict)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)