from flask import Flask, request, jsonify
from flask_cors import CORS 
from ultralytics import YOLO
import cv2
import numpy as np
import os

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app)  # AJOUTE CETTE LIGNE

# Chargement du modèle YOLOv8 entraîné
model = YOLO("best.pt")


# Route GET pour vérifier que l'API est en ligne
@app.route("/", methods=["GET"])
def home():
    return "✅ API YOLOv8 en ligne. Utilisez /predict pour envoyer une image."


# Route POST pour faire une prédiction sur une image
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8),
                         cv2.IMREAD_COLOR)

    # Prédiction avec le modèle YOLO
    results = model(image)

    predictions = []
    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if conf > 0.3:
                predictions.append({
                    "class": cls,
                    "confidence": round(conf, 2)
                })

    return jsonify({"results": predictions})


# Bloc indispensable pour le déploiement sur Render ou Replit
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render/Replit fournit le port
    app.run(host="0.0.0.0", port=port)



# from flask import Flask, request, jsonify
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import os

# # Initialisation de l'application Flask
# app = Flask(__name__)

# # Chargement du modèle YOLOv8 entraîné
# model = YOLO("best.pt")

# # Route GET pour vérifier que l'API est en ligne
# @app.route("/", methods=["GET"])
# def home():
#     return "✅ API YOLOv8 en ligne. Utilisez /predict pour envoyer une image."

# # Route POST pour faire une prédiction sur une image
# @app.route("/predict", methods=["POST"])
# def predict():
#     file = request.files['image']
#     image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

#     # Prédiction avec le modèle YOLO
#     results = model(image)

#     predictions = []
#     for r in results:
#         for box in r.boxes:
#             cls = model.names[int(box.cls[0])]
#             conf = float(box.conf[0])
#             if conf > 0.3:
#                 predictions.append({
#                     "class": cls,
#                     "confidence": round(conf, 2)
#                 })

#     return jsonify({"results": predictions})

# # Bloc indispensable pour le déploiement sur Render
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))  # Render fournit le port
#     app.run(host="0.0.0.0", port=port)
