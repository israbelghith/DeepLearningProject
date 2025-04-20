from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64
import io

app = Flask(__name__)

# Load the TensorFlow model (either .h5 or SavedModel format)
# For .h5 file:
model = tf.keras.models.load_model("model/cnn_emotion_detection.h5")

# For SavedModel (comment above and uncomment below):
# model = tf.keras.models.load_model("model/emotion_model")

# Preprocessing function (customize based on your training pipeline)
def preprocess_image(image_bytes):
    # Decode image from base64
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize and normalize
    img = cv2.resize(img, (48, 48))  # example size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # if grayscale model
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # (48, 48, 1)
    img = np.expand_dims(img, axis=0)   # (1, 48, 48, 1)

    return img

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    try:
        processed_img = preprocess_image(image_bytes)
        prediction = model.predict(processed_img)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        return jsonify({
            "emotion": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

