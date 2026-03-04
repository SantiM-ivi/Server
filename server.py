from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image as Image
import requests
import os
import base64
import io

app = Flask(__name__)

# Permitir hasta 16 MB por petición
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- CARGA DEL MODELO ---
# Cargamos el modelo MobileNet V2 de TensorFlow Hub
print("Cargando modelo de IA...")
model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5")

# Descargamos las etiquetas de ImageNet
labels_path = tf.keras.utils.get_file(
    "ImageNetLabels.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
)
labels = open(labels_path).read().splitlines()
print("Modelo listo para clasificar.")

# --- ENDPOINTS ---

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
        # 1. DECODIFICAR BASE64 A IMAGEN
        # Eliminamos encabezados si Godot los enviara (data:image/jpeg;base64,...)
        img_b64 = data["imagen"]
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]
            
        image_bytes = base64.b64decode(img_b64)
        img_file = io.BytesIO(image_bytes)
        
        # 2. PREPROCESAMIENTO PARA TENSORFLOW
        img = Image.open(img_file).convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = img_array.astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # 3. PREDICCIÓN
        predictions = model(img_array)
        predicted_class = np.argmax(predictions[0])
        etiqueta = labels[predicted_class]
        
        print(f"IA detectó: {etiqueta}")

        # 4. CONSULTAR MET MUSEUM
        # Buscamos el objeto por la etiqueta detectada
        search_url = f"https://collectionapi.metmuseum.org/public/collection/v1/search?q={etiqueta}"
        search_response = requests.get(search_url).json()

        result = {"query": etiqueta}
        
        if search_response.get("total", 0) > 0:
            # Tomamos el primer ID de objeto encontrado
            object_id = search_response["objectIDs"][0]
            object_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
            object_data = requests.get(object_url).json()
            
            # Agregamos la información del museo al resultado
            result.update({
                "title": object_data.get("title", "Sin título"),
                "artist": object_data.get("artistDisplayName", "Artista desconocido"),
                "date": object_data.get("objectDate", "Fecha desconocida"),
                "department": object_data.get("department", ""),
                "url": object_data.get("objectURL", "")
            })
        
        return jsonify(result)

    except Exception as e:
        print(f"Error procesando la imagen: {e}")
        return jsonify({"error": str(e)}), 500

# --- INICIO DEL SERVIDOR ---
if __name__ == "__main__":
    # Render usa la variable de entorno PORT
    port = int(os.environ.get("PORT", 5000))
    # Importante usar 0.0.0.0 para que sea accesible externamente
    app.run(host="0.0.0.0", port=port)





