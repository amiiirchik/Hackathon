from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# модель EfficientNet
model = EfficientNetB4(weights='imagenet', include_top=False, pooling='avg')

# функция для извлечения признаков из изображения
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    preprocessed_img = preprocess_input(img_array)
    features = model.predict(preprocessed_img).flatten()
    return features

# эталонные изображения
def load_reference_images():
    reference_images = {}
    for filename in os.listdir("./static/reference_images"):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join("./static/reference_images", filename)
            features = extract_features(img_path)
            reference_images[filename] = features
    return reference_images

# сравнение изображений
def compare_images(query_image, reference_images):
    query_features = extract_features(query_image)
    similarities = []
    for ref_filename, ref_features in reference_images.items():
        similarity = cosine_similarity(query_features.reshape(1, -1), ref_features.reshape(1, -1))[0][0]
        similarities.append((similarity, f'/static/reference_images/{ref_filename}'))
    similarities.sort(reverse=True)
    return similarities[:10]

# инициализация приложения
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['STATIC_FOLDER'] = './static/reference_images'

# Эталонные изображения при старте приложения
REFERENCE_IMAGES = load_reference_images()

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        similar_images = compare_images(filepath, REFERENCE_IMAGES)
        
        return jsonify({
            'similar_images': [url for _, url in similar_images],
        }), 200
    else:
        return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    app.run(debug=True)