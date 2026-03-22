from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import tensorflow as tf

app = Flask(__name__)

# Memory optimization - Load model ONCE
print("🔄 Loading brain_model.h5...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

model = load_model("brain_model.h5")
print("✅ Model loaded successfully!")

@app.route('/', methods=['GET'])
def home():
    return send_from_directory('.', 'index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files['file']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (150, 150))
        image = np.expand_dims(image/255.0, axis=0)
        
        prediction = model.predict(image, verbose=0)[0][0]
        result = "TUMOR" if prediction > 0.5 else "NO TUMOR"
        confidence = float(prediction if prediction > 0.5 else 1-prediction)
        
        return jsonify({
            "result": result,
            "confidence": f"{confidence:.1%}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
