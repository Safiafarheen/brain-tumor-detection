from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("brain_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    filepath = "temp.jpg"
    file.save(filepath)

    img = cv2.imread(filepath)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.reshape(img, (1, 64, 64, 3))

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        result = "Tumor Detected"
    else:
        result = "No Tumor"

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)