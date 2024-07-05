# Import necessary libraries
import os
from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

# Initialize Flask application
app = Flask(__name__)

# Load pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')

# Define a function to preprocess images
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((299, 299))  # Resize to match InceptionV3 input size
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Define a route to render the upload form
@app.route('/')
def upload_form():
    return render_template('upload.html')

# Define a route to handle image uploads
@app.route('/upload', methods=['POST'])
def upload_image():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        image_path = os.path.join('uploads', uploaded_file.filename)
        uploaded_file.save(image_path)

        # Preprocess the uploaded image
        img = preprocess_image(image_path)

        # Make predictions using the InceptionV3 model
        predictions = model.predict(img)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        # Return the top predictions
        top_predictions = [{'label': label, 'confidence': confidence} for (_, label, confidence) in decoded_predictions]

        return render_template('result.html', predictions=top_predictions)

# Run the application
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)



