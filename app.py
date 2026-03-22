from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load model globally (once at startup)
model = load_model("brain_model.h5")

@app.route('/', methods=['GET'])
def home():
    """Serve professional UI"""
    return send_from_directory('.', 'index.html')

@app.route("/predict", methods=["POST"])
def predict():
    """AI Brain Tumor Detection API"""
    try:
        # Get uploaded file
        file = request.files["file"]
        filepath = "temp.jpg"
        file.save(filepath)

        # Preprocess image
        img = cv2.imread(filepath)
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        img = np.reshape(img, (1, 64, 64, 3))

        # AI Prediction
        prediction = model.predict(img, verbose=0)

        # Binary classification
        if prediction[0][0] > 0.5:
            result = "Tumor Detected"
            confidence = f"{float(prediction[0][0])*100:.1f}%"
        else:
            result = "No Tumor"
            confidence = f"{float(1-prediction[0][0])*100:.1f}%"

        # Cleanup
        os.remove(filepath)
        
        return jsonify({
            "result": result,
            "confidence": confidence,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({
            "result": "Error",
            "error": str(e),
            "status": "error"
        })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
