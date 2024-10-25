from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask_cors import CORS  # Import CORS
from PIL import Image
import numpy as np
import io
import cv2

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the .h5 model
model = load_model('server/ecg_classification_model.h5')  # Update this path as needed

# Define allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the image before sending it to the model
def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get contours
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y+h, x:x+w]
    else:
        # If no contours are found, return the original image
        cropped_image = image

    # Resize to the target size for the model input
    target_width, target_height = 960, 540  # Adjust based on your model's expected input
    resized_image = cv2.resize(cropped_image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    # Convert to array and normalize
    image_array = img_to_array(resized_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            # Read the image file into an OpenCV image
            image = Image.open(io.BytesIO(file.read()))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Preprocess the image
            preprocessed_image = preprocess_image(image)

            # Predict using the model
            predictions = model.predict(preprocessed_image)
            class_index = np.argmax(predictions[0])
            
            # Define class labels (update as per your requirement)
            class_labels = ['Myocardial', 'Abnormal heartbeat', 'History of MI', 'Normal Person']
            predicted_class = class_labels[class_index]

            return jsonify({"prediction": predicted_class})

        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)