from flask import Flask, request, jsonify
from flask_cors import CORS
import base64, os, json, io, uuid, numpy as np
from PIL import Image
import onnxruntime as ort
from sqlalchemy import create_engine, text
import requests as http_requests
import traceback
import gc

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN ---
DB_URL = "postgresql://postgres.oiijjpwfzgoprmjbsjrk:facugodot2026@aws-1-us-east-1.pooler.supabase.com:6543/postgres"
MODEL_PATH = "modelo_museo.onnx"
SIMILARITY_THRESHOLD = 0.75

# Supabase Storage
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_API_KEY = os.environ.get("SUPABASE_API_KEY", "")
STORAGE_BUCKET = "fotos-piezas"

engine = create_engine(DB_URL, pool_pre_ping=True)

# Carga de ONNX optimizada para baja RAM
session = None
try:
    if os.path.exists(MODEL_PATH):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
        model_status = f"✅ ONNX Cargado: {MODEL_PATH}"
    else:
        model_status = "❌ ARCHIVO .ONNX NO ENCONTRADO"
except Exception as e:
    model_status = f"❌ ERROR ONNX: {str(e)}"

# --- FUNCIONES DE APOYO ---
def get_embedding(img_bytes):
    if not session: return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        arr = arr.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
        
        input_name = session.get_inputs()[0].name
        embedding = session.run(None, {input_name: arr})[0][0]
        
        norm = np.linalg.norm(embedding)
        vec = (embedding / norm).tolist() if norm > 0 else None
        
        del img, arr
        gc.collect()
        return vec
    except Exception as e:
        print(f"Error IA: {e}")
        return None

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
        if r.status_code in (200, 201):
            return f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{path}"
        return None
    except: return None

# --- RUTAS ---

@app.route('/')
def home():
    return f"🚀 Servidor Smartify Activo | {model_status}", 200

# RUTA PARA GODOT (ESCANEO)
@app.route('/clasificar', methods=['POST'])
def clasificar():
    try:
        data = request.get_json()
        img_bytes = base64.b64decode(data["imagen"].split(",")[1] if "," in data["imagen"] else data["imagen"])
        vec = get_embedding(img_bytes)
        
        if not vec: return jsonify({"error": "Error generando vector"}), 500
        print(f"DEBUG: Buscando vector de tamaño {len(vec)}")

        with engine.connect() as conn:
            query = text("""
                SELECT nombre, cultura, epoca, material, ubicacion, resumen, fotos_urls[1] as foto,
                1 - (embedding <=> :v::vector) as sim
                FROM piezas WHERE embedding IS NOT NULL ORDER BY embedding <=> :v::vector LIMIT 1
            """)
            res = conn.execute(query, {"v": str(vec)}).mappings().first()
            
        if res and res['sim'] >= SIMILARITY_THRESHOLD:
            return jsonify({
                "match": True, "nombre": res['nombre'], "cultura": res['cultura'], "epoca": res['epoca'],
                "material": res['material'], "ubicacion": res['ubicacion'], "resumen": res['resumen'],
                "foto_url": res['foto'], "confianza": round(float(res['sim']) * 100, 2)
            }), 200
        return jsonify({"match": False, "confianza": round(float(res['sim']) * 100, 2) if res else 0}), 404
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# RUTA PARA PANEL WEB (SUBIDA COMPLETA)
@app.route('/web/subir_completo', methods=['POST'])
def subir_pieza_museo():
    try:
        meta = json.loads(request.form.get('metadata'))
        fotos = request.files.getlist('fotos')
        fotos_validas = [f for f in fotos if f.filename != '']
        if not fotos_validas: return jsonify({"error": "Falta imagen"}), 400
        
        fotos_bytes = [f.read() for f in fotos_validas]
        # Generamos embedding de la primera foto
        emb = get_embedding(fotos_bytes[0])
        
        # Subir fotos a Storage
        urls = []
        for i, fb in enumerate(fotos_bytes):
            u = subir_foto_storage(fb, meta['nombre'], i)
            if u: urls.append(u)
        
        with engine.connect() as conn:
            conn.execute(text('''
                INSERT INTO piezas (museo_id, nombre, cultura, epoca, material, ubicacion, resumen, embedding, fotos_urls)
                VALUES (:m, :n, :c, :e, :mat, :u, :r, :emb, :urls)
                ON CONFLICT (nombre) DO UPDATE SET 
                    embedding=EXCLUDED.embedding, fotos_urls=EXCLUDED.fotos_urls,
                    cultura=EXCLUDED.cultura, epoca=EXCLUDED.epoca, material=EXCLUDED.material,
                    ubicacion=EXCLUDED.ubicacion, resumen=EXCLUDED.resumen
            '''), {
                "m": meta['museo'], "n": meta['nombre'], "c": meta['cultura'], "e": meta['epoca'], 
                "mat": meta['material'], "u": meta['ubicacion'], "r": meta['resumen'], 
                "emb": str(emb), "urls": urls
            })
            conn.commit()
        return jsonify({"status": "success"}), 201
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# RUTA EDITAR (MANTENIDA)
@app.route("/web/editar/<int:id>", methods=["POST"])
def editar_pieza(id):
    try:
        meta = json.loads(request.form.get('metadata'))
        with engine.connect() as conn:
            conn.execute(text('''
                UPDATE piezas SET museo_id=:m, nombre=:n, cultura=:c, epoca=:e,
                material=:mat, ubicacion=:u, resumen=:r WHERE id=:id
            '''), {"m": meta['museo'], "n": meta['nombre'], "c": meta['cultura'], "e": meta['epoca'], 
                   "mat": meta['material'], "u": meta['ubicacion'], "r": meta['resumen'], "id": id})
            conn.commit()
        return jsonify({"status": "updated"}), 200
    except Exception as e: return jsonify({"error": str(e)}), 500

# RUTA LISTA (MANTENIDA)
@app.route("/web/lista", methods=["GET"])
def listar_piezas():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT id, nombre, museo_id FROM piezas ORDER BY id DESC"))
            return jsonify([dict(r) for r in result.mappings().all()])
    except: return jsonify([])

# RUTA BORRAR (MANTENIDA)
@app.route("/web/borrar/<int:id>", methods=["DELETE"])
def borrar_pieza(id):
    try:
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM piezas WHERE id=:id"), {"id": id})
            conn.commit()
        return jsonify({"status": "deleted"}), 200
    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
