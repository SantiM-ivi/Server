from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import json
import io
import uuid
import numpy as np
from PIL import Image
import onnxruntime as ort
from sqlalchemy import create_engine, text
import requests as http_requests

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
DB_URL        = "postgresql://postgres.oiijjpwfzgoprmjbsjrk:facugodot2026@aws-1-us-east-1.pooler.supabase.com:6543/postgres"
MODEL_PATH    = "modelo_museo.onnx"
IMAGE_SIZE    = 224
EMBEDDING_DIM = 128
SIMILARITY_THRESHOLD = 0.82

# Supabase Storage — estos valores están en tu Dashboard → Settings → API
SUPABASE_URL     = os.environ.get("SUPABASE_URL", "")       # https://xxxx.supabase.co
SUPABASE_API_KEY = os.environ.get("SUPABASE_API_KEY", "")   # service_role key (no la anon)
STORAGE_BUCKET   = "fotos-piezas"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------------------------------------------------------------------------
# MODELO ONNX
# ---------------------------------------------------------------------------
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"'{MODEL_PATH}' no encontrado.")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])

try:
    modelo = cargar_modelo()
    print("✅ Modelo ONNX cargado.")
except FileNotFoundError as e:
    print(f"⚠️ {e}")
    modelo = None

# ---------------------------------------------------------------------------
# BASE DE DATOS
# fotos_urls: array de TEXT con las URLs públicas de Supabase Storage
# ---------------------------------------------------------------------------
engine = create_engine(DB_URL, pool_pre_ping=True, connect_args={'connect_timeout': 15})

def init_db():
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(f'''
                CREATE TABLE IF NOT EXISTS piezas (
                    id             SERIAL PRIMARY KEY,
                    museo_id       TEXT,
                    nombre         TEXT UNIQUE,
                    cultura        TEXT,
                    epoca          TEXT,
                    material       TEXT,
                    ubicacion      TEXT,
                    resumen        TEXT,
                    embedding      vector({EMBEDDING_DIM}),
                    fotos_urls     TEXT[],
                    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''))
            # Migración segura si la tabla ya existía sin fotos_urls
            conn.execute(text("ALTER TABLE piezas ADD COLUMN IF NOT EXISTS fotos_urls TEXT[]"))
            conn.execute(text(f'''
                CREATE INDEX IF NOT EXISTS idx_piezas_embedding
                ON piezas USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            '''))
            conn.commit()
            print("✅ Supabase DB + pgvector listo.")
    except Exception as e:
        print(f"⚠️ Error init_db: {e}")

init_db()

# ---------------------------------------------------------------------------
# SUPABASE STORAGE HELPERS
# ---------------------------------------------------------------------------
def subir_foto_storage(imagen_bytes: bytes, pieza_nombre: str, indice: int) -> str | None:
    """
    Sube una foto a Supabase Storage.
    Path: fotos-piezas/{pieza_nombre}/{uuid}.jpg
    Retorna la URL pública o None si falla.
    """
    if not SUPABASE_URL or not SUPABASE_API_KEY:
        print("⚠️ SUPABASE_URL o SUPABASE_API_KEY no configurados.")
        return None
    try:
        # Normalizar imagen antes de guardar
        img = Image.open(io.BytesIO(imagen_bytes)).convert('RGB')
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        buf.seek(0)

        # Nombre único para evitar colisiones
        slug = pieza_nombre.lower().replace(' ', '_').replace('/', '-')[:40]
        path = f"{slug}/{indice}_{uuid.uuid4().hex[:8]}.jpg"

        url = f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{path}"
        headers = {
            "Authorization": f"Bearer {SUPABASE_API_KEY}",
            "Content-Type": "image/jpeg",
            "x-upsert": "true"
        }
        r = http_requests.post(url, headers=headers, data=buf.getvalue(), timeout=20)

        if r.status_code in (200, 201):
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{path}"
            return public_url
        else:
            print(f"⚠️ Storage error {r.status_code}: {r.text}")
            return None
    except Exception as e:
        print(f"⚠️ Error subiendo foto a Storage: {e}")
        return None

# ---------------------------------------------------------------------------
# EMBEDDING HELPERS
# ---------------------------------------------------------------------------
def preprocesar(imagen_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(imagen_bytes)).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)

def calcular_embedding(imagen_bytes: bytes) -> list[float] | None:
    if modelo is None:
        return None
    try:
        salida = modelo.run(["embedding"], {"imagen": preprocesar(imagen_bytes)})
        vec    = salida[0][0].astype(np.float32)
        norma  = np.linalg.norm(vec)
        return (vec / norma).tolist() if norma > 0 else None
    except Exception as e:
        print(f"⚠️ Error inferencia: {e}")
        return None

def calcular_embedding_promedio(fotos_bytes: list[bytes]) -> list[float] | None:
    embeddings = [calcular_embedding(b) for b in fotos_bytes if b]
    embeddings = [e for e in embeddings if e is not None]
    if not embeddings:
        return None
    prom  = np.mean(embeddings, axis=0)
    norma = np.linalg.norm(prom)
    return (prom / norma).tolist() if norma > 0 else embeddings[0]

def vec_a_str(embedding: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------

@app.route('/')
def home():
    estado = "✅ cargado" if modelo else "❌ no encontrado"
    return f"Servidor Activo | Modelo ONNX: {estado}", 200

@app.route("/web/subir_completo", methods=["POST"])
def subir_pieza_museo():
    """
    Crea una pieza nueva:
    1. Sube todas las fotos a Supabase Storage
    2. Calcula embedding promedio
    3. Guarda metadata + URLs + embedding en la DB
    """
    if modelo is None:
        return jsonify({"error": "Modelo ONNX no disponible."}), 503
    try:
        meta  = json.loads(request.form.get('metadata'))
        fotos = request.files.getlist('fotos')
        fotos_validas = [f for f in fotos if f.filename != '']

        if not fotos_validas:
            return jsonify({"error": "Se requiere al menos una foto."}), 400

        # Leer todos los bytes primero (los streams se consumen una sola vez)
        fotos_bytes = [f.read() for f in fotos_validas]

        # Subir a Storage en paralelo (secuencial por simplicidad)
        urls = []
        for i, fb in enumerate(fotos_bytes):
            url = subir_foto_storage(fb, meta['nombre'], i)
            if url:
                urls.append(url)

        # Calcular embedding
        embedding = calcular_embedding_promedio(fotos_bytes)
        if embedding is None:
            return jsonify({"error": "No se pudo procesar las fotos."}), 400

        with engine.connect() as conn:
            conn.execute(text('''
                INSERT INTO piezas
                    (museo_id, nombre, cultura, epoca, material, ubicacion, resumen, embedding, fotos_urls)
                VALUES (:m, :n, :c, :e, :mat, :u, :r, :emb, :urls)
                ON CONFLICT (nombre) DO UPDATE SET
                    cultura=EXCLUDED.cultura, epoca=EXCLUDED.epoca,
                    material=EXCLUDED.material, ubicacion=EXCLUDED.ubicacion,
                    resumen=EXCLUDED.resumen, embedding=EXCLUDED.embedding,
                    fotos_urls=EXCLUDED.fotos_urls
            '''), {
                "m": meta['museo'], "n": meta['nombre'], "c": meta['cultura'],
                "e": meta['epoca'],  "mat": meta['material'], "u": meta['ubicacion'],
                "r": meta['resumen'], "emb": vec_a_str(embedding), "urls": urls
            })
            conn.commit()

        return jsonify({
            "status": "success",
            "fotos_procesadas": len(fotos_bytes),
            "fotos_guardadas_storage": len(urls)
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/web/editar/<int:id>", methods=["POST"])
def editar_pieza(id):
    """
    Edita metadata de una pieza.
    Si vienen fotos nuevas: sube a Storage, recalcula embedding, reemplaza URLs.
    Si no vienen fotos: solo actualiza los campos de texto.
    """
    try:
        meta        = json.loads(request.form.get('metadata'))
        fotos       = request.files.getlist('fotos')
        fotos_validas = [f for f in fotos if f.filename != '']

        if fotos_validas and modelo is not None:
            fotos_bytes = [f.read() for f in fotos_validas]

            urls = []
            for i, fb in enumerate(fotos_bytes):
                url = subir_foto_storage(fb, meta['nombre'], i)
                if url:
                    urls.append(url)

            embedding = calcular_embedding_promedio(fotos_bytes)
            if embedding is None:
                return jsonify({"error": "No se pudo procesar las fotos nuevas."}), 400

            with engine.connect() as conn:
                conn.execute(text('''
                    UPDATE piezas SET
                        museo_id=:m, nombre=:n, cultura=:c, epoca=:e,
                        material=:mat, ubicacion=:u, resumen=:r,
                        embedding=:emb, fotos_urls=:urls
                    WHERE id=:id
                '''), {
                    "m": meta['museo'], "n": meta['nombre'], "c": meta['cultura'],
                    "e": meta['epoca'],  "mat": meta['material'], "u": meta['ubicacion'],
                    "r": meta['resumen'], "emb": vec_a_str(embedding),
                    "urls": urls, "id": id
                })
                conn.commit()
            return jsonify({"status": "updated", "embedding_actualizado": True, "fotos_guardadas": len(urls)}), 200

        else:
            # Solo actualizar metadata, mantener embedding y fotos anteriores
            with engine.connect() as conn:
                conn.execute(text('''
                    UPDATE piezas SET
                        museo_id=:m, nombre=:n, cultura=:c, epoca=:e,
                        material=:mat, ubicacion=:u, resumen=:r
                    WHERE id=:id
                '''), {
                    "m": meta['museo'], "n": meta['nombre'], "c": meta['cultura'],
                    "e": meta['epoca'],  "mat": meta['material'], "u": meta['ubicacion'],
                    "r": meta['resumen'], "id": id
                })
                conn.commit()
            return jsonify({"status": "updated", "embedding_actualizado": False}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/web/lista", methods=["GET"])
def listar_piezas():
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT id, nombre, museo_id FROM piezas ORDER BY id DESC"
            ))
            return jsonify([dict(r) for r in result.mappings().all()])
    except:
        return jsonify([])

@app.route("/web/obtener/<int:id>", methods=["GET"])
def obtener_pieza(id):
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM piezas WHERE id=:id"), {"id": id})
            pieza  = result.mappings().first()
        if pieza:
            row = dict(pieza)
            row.pop('embedding', None)
            return jsonify(row)
        return {}, 404
    except:
        return jsonify({}), 500

@app.route("/web/borrar/<int:id>", methods=["DELETE"])
def borrar_pieza(id):
    try:
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM piezas WHERE id=:id"), {"id": id})
            conn.commit()
        return jsonify({"status": "deleted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/clasificar", methods=["POST"])
def clasificar():
    if modelo is None:
        return jsonify({"error": "Modelo ONNX no disponible."}), 503

    data = request.get_json()
    if not data or "imagen" not in data:
        return jsonify({"error": "No hay imagen"}), 400

    try:
        img_b64   = data["imagen"].split(",")[1] if "," in data["imagen"] else data["imagen"]
        img_bytes = base64.b64decode(img_b64)

        embedding = calcular_embedding(img_bytes)
        if embedding is None:
            return jsonify({"error": "No se pudo procesar la imagen."}), 400

        with engine.connect() as conn:
            result = conn.execute(text('''
                SELECT nombre, cultura, epoca, material, ubicacion, resumen,
                       1 - (embedding <=> :vec::vector) AS similitud
                FROM piezas
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> :vec::vector
                LIMIT 1
            '''), {"vec": vec_a_str(embedding)})
            row = result.mappings().first()

        if row and float(row['similitud']) >= SIMILARITY_THRESHOLD:
            print(f"✅ Match: '{row['nombre']}' | Similitud: {row['similitud']:.4f}")
            return jsonify({
                "nombre":    row['nombre'],
                "cultura":   row['cultura'],
                "epoca":     row['epoca'],
                "material":  row['material'],
                "ubicacion": row['ubicacion'],
                "resumen":   row['resumen']
            })

        print(f"❌ Sin match | Similitud: {row['similitud']:.4f if row else 'N/A'}")
        return jsonify({"error": "Pieza no reconocida"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
