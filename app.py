from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

app = Flask(__name__)
model = YOLO("best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model(image)
    
    predictions = []
    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if conf > 0.3:
                predictions.append({"class": cls, "confidence": round(conf, 2)})

    return jsonify({"results": predictions})
