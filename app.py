from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)
# Load your trained model
model = tf.keras.models.load_model("pneumonia_mobilenetv2.keras")

@app.route("/")
def index():
    # Serve the index.html file from the root folder
    return send_from_directory(".", "index.html")

@app.route("/style.css")
def style():
    # Serve the CSS file from the root folder
    return send_from_directory(".", "style.css")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file:
        # Read the file bytes and wrap in BytesIO
        file_bytes = file.read()
        file_stream = BytesIO(file_bytes)
        # Open the image, force conversion to RGB, and resize to 224x224
        img = Image.open(file_stream).convert("RGB").resize((224, 224))
        # Convert image to array and normalize to [0, 1]
        img_array = np.array(img) / 255.0
        # Ensure the shape is (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        # Run prediction
        prob = model.predict(img_array)[0][0]
        if prob > 0.7:
            return jsonify({"prediction": "PNEUMONIA", "confidence": float(prob)})
        else:
            return jsonify({"prediction": "NORMAL", "confidence": float(1.0 - prob)})
    return jsonify({"error": "File upload failed"}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
