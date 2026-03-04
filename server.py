from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image as Image
import requests
import os
import base64
import io
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- CARGA DEL MODELO ---
print("Cargando IA de alto rendimiento...")
model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5")
labels_path = tf.keras.utils.get_file("ImageNetLabels.txt", "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt")
labels = open(labels_path).read().splitlines()

# --- FUNCIONES DE BÚSQUEDA ---

def fetch_info(query):
    """Consulta Wikipedia y Museos para una etiqueta específica."""
    resultado = {"etiqueta": query, "detalles": None, "museo": None}
    
    # Intentar Wikipedia
    try:
        wiki_url = f"https://es.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        res = requests.get(wiki_url, timeout=2).json()
        if "extract" in res:
            resultado["detalles"] = res["extract"]
    except: pass

    # Intentar Met Museum
    try:
        m_url = f"https://collectionapi.metmuseum.org/public/collection/v1/search?q={query}"
        m_res = requests.get(m_url, timeout=2).json()
        if m_res.get("total", 0) > 0:
            obj_id = m_res["objectIDs"][0]
            o_data = requests.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}", timeout=2).json()
            resultado["museo"] = {
                "obra": o_data.get("title"),
                "artista": o_data.get("artistDisplayName"),
                "fecha": o_data.get("objectDate")
            }
    except: pass
    
    return resultado

# --- ENDPOINT ---

@app.route("/clasificar", methods=["POST"])
def clasificar():
    data = request.get_json()
    if not data or "imagen" not in data:
        return jsonify({"error": "No image"}), 400

    try:
        # 1. Procesar Imagen
        img_b64 = data["imagen"].split(",")[1] if "," in data["imagen"] else data["imagen"]
        img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB").resize((224, 224))
        img_array = np.array(img).astype(np.float32)[np.newaxis, ...] / 255.0

        # 2. DETECCIÓN MULTI-ETIQUETA (Top 5)
        preds = model(img_array)
        # Obtenemos los índices de las 5 mejores predicciones
        top_5_indices = np.argsort(preds[0])[-5:][::-1]
        
        # Limpiamos las etiquetas (quitamos comas y espacios)
        top_queries = [labels[i].split(",")[0].strip() for i in top_5_indices]

        # 3. BÚSQUEDA EN PARALELO DE TODAS LAS ETIQUETAS
        # Buscamos info de las 5 cosas a la vez para no perder tiempo
        resultados_busqueda = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch_info, q) for q in top_queries]
            for f in futures:
                res = f.result()
                # Solo agregamos si encontramos algo de información real
                if res["detalles"] or res["museo"]:
                    resultados_busqueda.append(res)

        return jsonify({
            "predicciones_ia": top_queries,
            "busqueda_concreta": resultados_busqueda
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))






