from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
import openai

# Initialize the Flask app
app = Flask(__name__)

# Define the folder where uploaded files will be stored
UPLOAD_FOLDER = 'uploaded_files'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure the upload folder for Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the skin cancer detection model
def load_model():
    model = tf.keras.models.load_model('my_model.h5', compile=False)
    pass

# Predict function
def predict_class(image):
    # Preprocess the image
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Load the model
    model = load_model()

    # Predict the class of the image
    predictions = model.predict(image)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])

    # Map the predicted class index to the corresponding label
    label_mapping = {
  "0": "nevus",
  "1": "seborrheic keratosis",
  "2": "dermatofibroma",
  "3": "vascular lesion",
  "4": "pigmented benign keratosis",
  "5": "actinic keratosis",
  "6": "basal cell carcinoma",
  "7": "squamous cell carcinoma",
  "8": "melanoma"
}
    predicted_label = label_mapping.get(predicted_class_index, 'Unknown')

    return predicted_label

# OpenAI API Integration for treatment suggestions
def get_treatment_suggestions(predicted_class):
    # Set up your OpenAI API key
    openai.api_key = '<OPENAI API KEY>'

    # Customize the prompt based on the predicted class
    prompt = f"What is the recommended treatment for {predicted_class} skin cancer?"

    # Request treatment suggestions from OpenAI API
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )

    return response.choices[0].text.strip()

# Route for handling file upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the POST request has a file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        # Secure the filename to prevent malicious attacks
        filename = secure_filename(file.filename)
        # Save the file to the upload folder
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Read the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_image = cv2.imread(image_path)

        # Predict the class of the uploaded image
        predicted_class = predict_class(uploaded_image)

        # Get treatment suggestions from OpenAI API
        treatment_suggestions = get_treatment_suggestions(predicted_class)

        # Respond with JSON containing predicted class and treatment suggestions
        return jsonify({
            'predicted_class': predicted_class,
            'treatment_suggestions': treatment_suggestions
        }), 200

    return jsonify({'error': 'File type not allowed'}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

