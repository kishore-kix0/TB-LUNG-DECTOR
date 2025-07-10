import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained TB model
MODEL_PATH = 'tb_cnn_model.h5'
model = load_model(MODEL_PATH)

print('Model loaded. Visit http://127.0.0.1:5000/')

def predict_tb(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = "Tuberculosis" if prediction > 0.5 else "Normal"
    confidence = round(prediction * 100, 2) if prediction > 0.5 else round(100 - prediction * 100, 2)
    return label, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result, confidence = predict_tb(filepath)
    return render_template('result.html', user_image=filepath, result=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
