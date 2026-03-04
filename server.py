from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image as Image
import requests
import os
import base64
import io

# 1. DEFINIR LA APP (Esto debe ir antes de cualquier @app.route)
app = Flask(__name__)

# Permitir hasta 16 MB por petición
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 2. CARGAR EL MODELO (Se hace al iniciar el servidor)
print("Cargando modelo de IA...")
try:
    model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5")
    labels_path = tf.keras.utils.get_file(
        "ImageNetLabels.txt",
        "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    )
    labels = open(labels_path).read().splitlines()
    print("Modelo y etiquetas listos.")
except Exception as e:
    print(f"Error cargando el modelo: {e}")

# 3. DEFINIR LOS ENDPOINTS
@app.route("/")
def home():
    return "Servidor online 🚀 - Esperando peticiones en /clasificar"

@app.route("/clasificar", methods=["POST"])
def clasificar():
    # Obtener los datos del JSON enviado por Godot
    data = request.get_json()
    
    if not data or "imagen" not in data:
        return jsonify({"error": "No se recibió el campo 'imagen' en el JSON"}), 400

    try:
        # Decodificar Base64
        img_b64 = data["imagen"]
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]
            
        image_bytes = base64.b64decode(img_b64)
        img_file = io.BytesIO(image_bytes)
        
        # Preprocesamiento
        img = Image.open(img_file).convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = img_array.astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Predicción
        predictions = model(img_array)
        predicted_class = np.argmax(predictions[0])
        etiqueta = labels[predicted_class]
        
        print(f"IA detectó: {etiqueta}")

        # Consultar Met Museum
        search_url = f"https://collectionapi.metmuseum.org/public/collection/v1/search?q={etiqueta}"
        search_response = requests.get(search_url).json()

        result = {"query": etiqueta}
        
        if search_response.get("total", 0) > 0:
            object_id = search_response["objectIDs"][0]
            object_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
            object_data = requests.get(object_url).json()
            
            result.update({
                "title": object_data.get("title", "Sin título"),
                "artist": object_data.get("artistDisplayName", "Artista desconocido"),
                "date": object_data.get("objectDate", "Fecha desconocida"),
                "department": object_data.get("department", ""),
                "url": object_data.get("objectURL", "")
            })
        
        return jsonify(result)

    except Exception as e:
        print(f"Error en clasificar: {e}")
        return jsonify({"error": str(e)}), 500

# 4. EJECUCIÓN
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)






