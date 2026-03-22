from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import io
import base64
from PIL import Image

app = Flask(__name__)

# Load model globally (once at startup)
model = load_model("brain_model.h5")

@app.route('/', methods=['GET'])
def home():
    """Serve professional UI"""
    return send_from_directory('.', 'index.html')

@app.route("/predict", methods=["POST"])
def predict():
    """AI Brain Tumor Detection API - RENDER FIXED"""
    try:
        # Get uploaded file - NO DISK SAVE!
        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "No file"}), 400
        
        # Read image from memory (Render fix!)
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess (same as yours but PIL + cv2)
        img = np.array(image)
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        img = np.reshape(img, (1, 64, 64, 3))

        # AI Prediction (YOUR exact logic)
        prediction = model.predict(img, verbose=0)

        # YOUR exact result logic
        if prediction[0][0] > 0.5:
            result = "Tumor Detected"
            confidence = f"{float(prediction[0][0])*100:.1f}%"
        else:
            result = "No Tumor"
            confidence = f"{float(1-prediction[0][0])*100:.1f}%"

        # Convert image to base64 for preview (Render compatible!)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            "result": result,
            "confidence": confidence,
            "status": "success",
            "image_b64": f"data:image/jpeg;base64,{img_str}"
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
