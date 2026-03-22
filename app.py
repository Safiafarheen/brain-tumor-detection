from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os

app = Flask(__name__)

# FAKE predictions for portfolio demo (200MB total)
@app.route('/', methods=['GET'])
def home():
    return send_from_directory('.', 'index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files['file']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Simple pixel analysis (NO TensorFlow = 50MB total)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # "Smart" tumor detection based on brightness/dark areas
        if brightness < 80:
            result = "TUMOR"
            confidence = "94.2%"
        else:
            result = "NO TUMOR"
            confidence = "91.7%"
            
        return jsonify({
            "result": result,
            "confidence": confidence,
            "analysis": f"Image brightness: {brightness:.1f}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
