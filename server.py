from flask import Flask, request, jsonify
from flask_cors import CORS
import base64, os, json, io, uuid, numpy as np
from PIL import Image
import onnxruntime as ort
from sqlalchemy import create_engine, text
import requests as http_requests

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN ---
DB_URL = "postgresql://postgres.oiijjpwfzgoprmjbsjrk:facugodot2026@aws-1-us-east-1.pooler.supabase.com:6543/postgres"
MODEL_PATH = "modelo_ia.onnx" # Asegurate de que este archivo esté en tu GitHub
SIMILARITY_THRESHOLD = 0.78

# Supabase Storage
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_API_KEY = os.environ.get("SUPABASE_API_KEY", "")
STORAGE_BUCKET = "fotos-piezas"

# Cargamos ONNX (Mucho más liviano que TensorFlow para Render)
try:
    if os.path.exists(MODEL_PATH):
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        model_status = "✅ ONNX Activo (Ligero)"
    else:
        session = None
        model_status = "⚠️ Falta archivo .onnx"
except Exception as e:
    session = None
    model_status = f"❌ Error ONNX: {str(e)}"

engine = create_engine(DB_URL, pool_pre_ping=True)

# --- HELPERS ---
def get_embedding(img_bytes):
    if not session: return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        # Normalización estándar ImageNet
        arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        arr = arr.transpose(2, 0, 1)[np.newaxis, :]
        
        # Corremos la inferencia
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        embedding = session.run([output_name], {input_name: arr.astype(np.float32)})[0][0]
        
        norm = np.linalg.norm(embedding)
        return (embedding / norm).tolist() if norm > 0 else None
    except: return None

def subir_foto_storage(imagen_bytes, pieza_nombre, indice):
    if not SUPABASE_URL or not SUPABASE_API_KEY: return None
    try:
        slug = pieza_nombre.lower().replace(' ', '_')[:30]
        path = f"{slug}/{indice}_{uuid.uuid4().hex[:6]}.jpg"
        url = f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{path}"
        headers = {"Authorization": f"Bearer {SUPABASE_API_KEY}", "Content-Type": "image/jpeg", "x-upsert": "true"}
        r = http_requests.post(url, headers=headers, data=imagen_bytes, timeout=15)
        return f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{path}" if r.status_code in (200, 201) else None
    except: return None

# --- RUTAS ---
@app.route('/')
def home():
    return f"🚀 Servidor Activo (Modo Estable) | {model_status}", 200

@app.route('/web/subir_completo', methods=['POST'])
def subir_pieza_museo():
    try:
        meta = json.loads(request.form.get('metadata'))
        fotos = request.files.getlist('fotos')
        if not fotos: return jsonify({"error": "Falta imagen"}), 400
        fb = fotos[0].read()
        emb = get_embedding(fb)
        url = subir_foto_storage(fb, meta['nombre'], 0)
        with engine.connect() as conn:
            conn.execute(text('''
                INSERT INTO piezas (museo_id, nombre, cultura, epoca, material, ubicacion, resumen, embedding, fotos_urls)
                VALUES (:m, :n, :c, :e, :mat, :u, :r, :emb, :urls)
                ON CONFLICT (nombre) DO UPDATE SET embedding=EXCLUDED.embedding, fotos_urls=EXCLUDED.fotos_urls
            '''), {"m": meta['museo'], "n": meta['nombre'], "c": meta['cultura'], "e": meta['epoca'], 
                   "mat": meta['material'], "u": meta['ubicacion'], "r": meta['resumen'], 
                   "emb": str(emb), "urls": [url] if url else []})
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

@app.route('/clasificar', methods=['POST'])
def clasificar():
    try:
        # 1. Validar que llegue un JSON
        data = request.get_json(silent=True)
        if not data or "imagen" not in data:
            return jsonify({"error": "JSON invalido o falta el campo 'imagen'"}), 400

        # 2. Limpiar el Base64 (por si Godot envía el prefijo data:image/jpg;base64,)
        img_data = data["imagen"]
        if "," in img_data:
            img_data = img_data.split(",")[1]
        
        try:
            img_bytes = base64.b64decode(img_data)
        except Exception as e:
            return jsonify({"error": "No se pudo decodificar Base64"}), 400

        # 3. Obtener el Embedding (ADN Visual)
        vec = get_embedding(img_bytes)
        if not vec:
            return jsonify({"error": "La IA no pudo procesar la imagen"}), 400

        # 4. Buscar en la Base de Datos
        with engine.connect() as conn:
            query = text("""
                SELECT nombre, cultura, epoca, material, ubicacion, resumen, fotos_urls[1] as foto,
                1 - (embedding <=> :v::vector) as sim
                FROM piezas 
                WHERE embedding IS NOT NULL 
                ORDER BY embedding <=> :v::vector 
                LIMIT 1
            """)
            res = conn.execute(query, {"v": str(vec)}).mappings().first()

        # 5. Respuesta
        if res and res['sim'] >= SIMILARITY_THRESHOLD:
            return jsonify({
                "match": True,
                "nombre": res['nombre'],
                "cultura": res['cultura'],
                "epoca": res['epoca'],
                "material": res['material'],
                "ubicacion": res['ubicacion'],
                "resumen": res['resumen'],
                "foto_url": res['foto'],
                "confianza": round(float(res['sim']) * 100, 2)
            }), 200
        
        return jsonify({"match": False, "confianza": round(float(res['sim']) * 100, 2) if res else 0}), 404

    except Exception as e:
        print(f"Error critico: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
