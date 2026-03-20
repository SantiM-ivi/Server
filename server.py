from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from sqlalchemy import create_engine, text

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
DB_URL = "postgresql://postgres.oiijjpwfzgoprmjbsjrk:facugodot2026@aws-1-us-east-1.pooler.supabase.com:6543/postgres"
SIMILARITY_THRESHOLD = 0.78  # Ajusta este valor según las pruebas en el museo

# Cargamos el modelo extractor de características (sin la capa de clasificación)
# Esto convierte cualquier imagen en un vector de 1024 números.
model = tf.keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3), 
    include_top=False, 
    pooling='avg'
)

engine = create_engine(DB_URL, pool_pre_ping=True)

# ---------------------------------------------------------------------------
# PROCESAMIENTO DE IA
# ---------------------------------------------------------------------------
def get_embedding(img_bytes):
    try:
        # Preparamos la imagen para la red neuronal
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
        x = tf.keras.applications.mobilenet_v3.preprocess_input(np.array(img))
        
        # Obtenemos el vector de identidad de la imagen
        embedding = model.predict(np.expand_dims(x, axis=0), verbose=0)[0]
        
        # Normalizamos el vector (importante para la distancia coseno)
        norm = np.linalg.norm(embedding)
        return (embedding / norm).tolist() if norm > 0 else None
    except Exception as e:
        print(f"Error en IA: {e}")
        return None

# ---------------------------------------------------------------------------
# RUTA PRINCIPAL DE ESCANEO
# ---------------------------------------------------------------------------
@app.route('/clasificar', methods=['POST'])
def clasificar():
    try:
        data = request.get_json()
        if not data or "imagen" not in data:
            return jsonify({"error": "No se recibió ninguna imagen"}), 400

        # Decodificar Base64 de Godot
        img_b64 = data["imagen"].split(",")[1] if "," in data["imagen"] else data["imagen"]
        img_bytes = base64.b64decode(img_b64)
        
        # Generar ADN Visual
        vec = get_embedding(img_bytes)
        if not vec:
            return jsonify({"error": "No se pudo procesar la imagen"}), 400

        # Buscar en Supabase usando pgvector (<=> es distancia coseno)
        # Comparamos el vector enviado con los guardados en la DB
        with engine.connect() as conn:
            query = text("""
                SELECT nombre, cultura, epoca, material, ubicacion, resumen, fotos_urls[1] as foto,
                       1 - (embedding <=> :v::vector) as similitud
                FROM piezas 
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> :v::vector 
                LIMIT 1
            """)
            result = conn.execute(query, {"v": str(vec)}).mappings().first()

        # Respuesta estilo Smartify
        if result and result['similitud'] >= SIMILARITY_THRESHOLD:
            return jsonify({
                "match": True,
                "nombre": result['nombre'],
                "cultura": result['cultura'],
                "epoca": result['epoca'],
                "material": result['material'],
                "ubicacion": result['ubicacion'],
                "resumen": result['resumen'],
                "foto_url": result['foto'],
                "confianza": round(float(result['similitud']) * 100, 2)
            }), 200
        
        return jsonify({
            "match": False, 
            "error": "Pieza no reconocida",
            "score": float(result['similitud']) if result else 0
        }), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
