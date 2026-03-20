from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import json
import io
import uuid
import numpy as np
from PIL import Image
import tensorflow as tf
from sqlalchemy import create_engine, text
import requests as http_requests

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
DB_URL = "postgresql://postgres.oiijjpwfzgoprmjbsjrk:facugodot2026@aws-1-us-east-1.pooler.supabase.com:6543/postgres"
SIMILARITY_THRESHOLD = 0.78

# Supabase Storage
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_API_KEY = os.environ.get("SUPABASE_API_KEY", "")
STORAGE_BUCKET = "fotos-piezas"

# Cargamos el extractor de ADN Visual (MobileNetV3)
try:
    model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3), include_top=False, pooling='avg'
    )
    model_status = "✅ Modelo IA Cargado (MobileNetV3)"
except Exception as e:
    model = None
    model_status = f"❌ Error cargando modelo: {str(e)}"

engine = create_engine(DB_URL, pool_pre_ping=True)

# ---------------------------------------------------------------------------
# HELPERS IA Y STORAGE
# ---------------------------------------------------------------------------
def get_embedding(img_bytes):
    if not model: return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
        x = tf.keras.applications.mobilenet_v3.preprocess_input(np.array(img))
        embedding = model.predict(np.expand_dims(x, axis=0), verbose=0)[0]
        norm = np.linalg.norm(embedding)
        return (embedding / norm).tolist() if norm > 0 else None
    except: return None

def subir_foto_storage(imagen_bytes, pieza_nombre, indice):
    if not SUPABASE_URL or not SUPABASE_API_KEY: return None
    try:
        slug = pieza_nombre.lower().replace(' ', '_')[:30]
        path = f"{slug}/{indice}_{uuid.uuid4().hex[:6]}.jpg"
        url = f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{path}"
        headers = {
            "Authorization": f"Bearer {SUPABASE_API_KEY}",
            "Content-Type": "image/jpeg",
            "x-upsert": "true"
        }
        r = http_requests.post(url, headers=headers, data=imagen_bytes, timeout=15)
        return f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{path}" if r.status_code in (200, 201) else None
    except: return None

# ---------------------------------------------------------------------------
# RUTAS DE ESTADO Y PANEL WEB
# ---------------------------------------------------------------------------
@app.route('/')
def home():
    return f"🚀 Servidor Activo | {model_status}", 200

@app.route('/web/subir_completo', methods=['POST'])
def subir_pieza_museo():
    try:
        meta = json.loads(request.form.get('metadata'))
        fotos = request.files.getlist('fotos')
        if not fotos: return jsonify({"error": "Falta imagen"}), 400

        foto_principal = fotos[0].read()
        embedding = get_embedding(foto_principal)
        url = subir_foto_storage(foto_principal, meta['nombre'], 0)
        urls = [url] if url else []

        with engine.connect() as conn:
            conn.execute(text('''
                INSERT INTO piezas (museo_id, nombre, cultura, epoca, material, ubicacion, resumen, embedding, fotos_urls)
                VALUES (:m, :n, :c, :e, :mat, :u, :r, :emb, :urls)
                ON CONFLICT (nombre) DO UPDATE SET embedding=EXCLUDED.embedding, fotos_urls=EXCLUDED.fotos_urls
            '''), {
                "m": meta['museo'], "n": meta['nombre'], "c": meta['cultura'],
                "e": meta['epoca'], "mat": meta['material'], "u": meta['ubicacion'],
                "r": meta['resumen'], "emb": str(embedding), "urls": urls
            })
            conn.commit()
        return jsonify({"status": "success"}), 201
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/web/lista", methods=["GET"])
def listar_piezas():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT id, nombre, museo_id FROM piezas ORDER BY id DESC"))
        return jsonify([dict(r) for r in result.mappings().all()])

@app.route("/web/borrar/<int:id>", methods=["DELETE"])
def borrar_pieza(id):
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM piezas WHERE id=:id"), {"id": id})
        conn.commit()
    return jsonify({"status": "deleted"}), 200

# ---------------------------------------------------------------------------
# RUTA ESCANEO GODOT (SMARTIFY STYLE)
# ---------------------------------------------------------------------------
@app.route('/clasificar', methods=['POST'])
def clasificar():
    try:
        data = request.get_json()
        img_b64 = data["imagen"].split(",")[1] if "," in data["imagen"] else data["imagen"]
        img_bytes = base64.b64decode(img_b64)
        
        vec = get_embedding(img_bytes)
        if not vec: return jsonify({"error": "Error de IA"}), 400

        with engine.connect() as conn:
            query = text("""
                SELECT nombre, cultura, epoca, material, ubicacion, resumen, fotos_urls[1] as foto,
                       1 - (embedding <=> :v::vector) as similitud
                FROM piezas WHERE embedding IS NOT NULL
                ORDER BY embedding <=> :v::vector LIMIT 1
            """)
            result = conn.execute(query, {"v": str(vec)}).mappings().first()

        if result and result['similitud'] >= SIMILARITY_THRESHOLD:
            return jsonify({
                "match": True, "nombre": result['nombre'], "cultura": result['cultura'],
                "epoca": result['epoca'], "material": result['material'],
                "ubicacion": result['ubicacion'], "resumen": result['resumen'],
                "foto_url": result['foto'], "confianza": round(float(result['similitud']) * 100, 2)
            }), 200
        return jsonify({"match": False, "error": "No reconocida"}), 404
    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
