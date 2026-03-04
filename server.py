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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- CARGA DEL MODELO ---
print("Cargando IA...")
model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5")
labels_path = tf.keras.utils.get_file("ImageNetLabels.txt", "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt")
labels = open(labels_path).read().splitlines()

# --- FUNCIONES DE BÚSQUEDA ---

def buscar_en_wikipedia(query):
    try:
        # Buscamos en Wikipedia en Español
        url = f"https://es.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            data = res.json()
            return {
                "titulo": data.get("title"),
                "descripcion": data.get("extract"),
                "wiki_url": data.get("content_urls", {}).get("desktop", {}).get("page")
            }
    except: return None
    return None

def buscar_en_museos(query):
    """ Intenta buscar en el Met Museum para info técnica """
    try:
        url_search = f"https://collectionapi.metmuseum.org/public/collection/v1/search?q={query}"
        res_search = requests.get(url_search, timeout=3).json()
        if res_search.get("total", 0) > 0:
            obj_id = res_search["objectIDs"][0]
            obj_data = requests.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}", timeout=3).json()
            return {
                "obra": obj_data.get("title"),
                "artista": obj_data.get("artistDisplayName"),
                "fecha": obj_data.get("objectDate"),
                "museo": "Metropolitan Museum of Art"
            }
    except: return None
    return None

# --- ENDPOINT PRINCIPAL ---

@app.route("/clasificar", methods=["POST"])
def clasificar():
    data = request.get_json()
    if not data or "imagen" not in data:
        return jsonify({"error": "No hay imagen"}), 400

    try:
        # 1. Procesar Imagen
        img_b64 = data["imagen"].split(",")[1] if "," in data["imagen"] else data["imagen"]
        img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB").resize((224, 224))
        img_array = np.array(img).astype(np.float32)[np.newaxis, ...] / 255.0

        # 2. IA Predicción
        predictions = model(img_array)
        etiqueta = labels[np.argmax(predictions[0])].split(",")[0]
        
        # 3. BÚSQUEDA MULTI-API (Wikipedia + Museos)
        wiki_info = buscar_en_wikipedia(etiqueta)
        museo_info = buscar_en_museos(etiqueta)

        # 4. Combinar Resultados
        respuesta = {
            "query": etiqueta,
            "wikipedia": wiki_info if wiki_info else {"descripcion": "No se encontró artículo detallado."},
            "museo": museo_info if museo_info else None
        }

        return jsonify(respuesta)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))






