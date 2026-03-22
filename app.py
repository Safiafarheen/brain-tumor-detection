from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

print("🔄 Loading brain_model.h5...")
model = load_model("brain_model.h5")
print("✅ Model loaded! Input shape:", model.input_shape)

@app.route('/', methods=['GET'])
def home():
    return send_from_directory('.', 'index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files['file']
        # FIX 1: Grayscale + Resize to model's expected 64x64
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (64, 64))  # Model expects 64x64
        image = image.reshape(1, 64, 64, 1)  # Add channel dim
        image = image / 255.0
        
        prediction = model.predict(image, verbose=0)[0][0]
        result = "TUMOR" if prediction > 0.5 else "NO TUMOR"
        confidence = float(prediction if prediction > 0.5 else 1-prediction)
        
        return jsonify({
            "result": result,
            "confidence": f"{confidence:.1%}",
            "input_shape": str(image.shape)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
