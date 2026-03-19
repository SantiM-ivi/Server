from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import json
import io
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.transform import resize as sk_resize
from skimage.color import rgb2gray
from scipy.spatial.distance import cosine
from sqlalchemy import create_engine, text

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN DE BASE DE DATOS ---
DB_URL = "postgresql://postgres.oiijjpwfzgoprmjbsjrk:facugodot2026@aws-1-us-east-1.pooler.supabase.com:6543/postgres"

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    connect_args={'connect_timeout': 15}
)

def init_db():
    try:
        with engine.connect() as conn:
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS piezas (
                    id SERIAL PRIMARY KEY,
                    museo_id TEXT,
                    nombre TEXT UNIQUE,
                    cultura TEXT,
                    epoca TEXT,
                    material TEXT,
                    ubicacion TEXT,
                    resumen TEXT,
                    hog_descriptor TEXT,
                    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''))
            # Migración segura: agrega la columna si la tabla ya existía sin ella
            conn.execute(text('''
                ALTER TABLE piezas
                ADD COLUMN IF NOT EXISTS hog_descriptor TEXT
            '''))
            conn.commit()
            print("✅ Base de Datos en Supabase lista.")
    except Exception as e:
        print(f"⚠️ Error de conexión inicial: {e}")

init_db()

# --- PARÁMETROS HOG ---
HOG_IMAGE_SIZE = (128, 128)
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": True,
}
# Distancia coseno: 0.0 = idéntico. Menor a este umbral = match válido.
HOG_THRESHOLD = 0.35

# --- FUNCIONES HOG ---

def calcular_hog(imagen_bytes: bytes) -> np.ndarray | None:
    try:
        img = Image.open(io.BytesIO(imagen_bytes)).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_gray = rgb2gray(img_array)
        img_resized = sk_resize(img_gray, HOG_IMAGE_SIZE, anti_aliasing=True)
        return hog(img_resized, **HOG_PARAMS)
    except Exception as e:
        print(f"⚠️ Error calculando HOG: {e}")
        return None

def descriptor_a_texto(descriptor: np.ndarray) -> str:
    return json.dumps(descriptor.tolist())

def texto_a_descriptor(texto: str) -> np.ndarray | None:
    try:
        return np.array(json.loads(texto), dtype=np.float32)
    except:
        return None

# --- RUTAS ---

@app.route('/')
def home():
    return "Servidor Activo (Supabase + HOG) 🚀", 200

@app.route("/web/subir_completo", methods=["POST"])
def subir_pieza_museo():
    try:
        meta = json.loads(request.form.get('metadata'))
        fotos = request.files.getlist('fotos')
        foto_bytes = fotos[0].read()

        descriptor = calcular_hog(foto_bytes)
        if descriptor is None:
            return jsonify({"error": "No se pudo calcular descriptor HOG"}), 400

        with engine.connect() as conn:
            conn.execute(text('''
                INSERT INTO piezas
                    (museo_id, nombre, cultura, epoca, material, ubicacion, resumen, hog_descriptor)
                VALUES
                    (:m, :n, :c, :e, :mat, :u, :r, :hog)
                ON CONFLICT (nombre) DO UPDATE SET
                    cultura        = EXCLUDED.cultura,
                    epoca          = EXCLUDED.epoca,
                    material       = EXCLUDED.material,
                    ubicacion      = EXCLUDED.ubicacion,
                    resumen        = EXCLUDED.resumen,
                    hog_descriptor = EXCLUDED.hog_descriptor
            '''), {
                "m": meta['museo'], "n": meta['nombre'], "c": meta['cultura'],
                "e": meta['epoca'], "mat": meta['material'], "u": meta['ubicacion'],
                "r": meta['resumen'], "hog": descriptor_a_texto(descriptor)
            })
            conn.commit()
        return jsonify({"status": "success"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/web/lista", methods=["GET"])
def listar_piezas():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT id, nombre, museo_id FROM piezas ORDER BY id DESC"))
            piezas = [dict(row) for row in result.mappings().all()]
        return jsonify(piezas)
    except:
        return jsonify([])

@app.route("/web/obtener/<int:id>", methods=["GET"])
def obtener_pieza(id):
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM piezas WHERE id = :id"), {"id": id})
            pieza = result.mappings().first()
        return jsonify(dict(pieza)) if pieza else ({}, 404)
    except:
        return jsonify({}), 500

@app.route("/web/borrar/<int:id>", methods=["DELETE"])
def borrar_pieza(id):
    try:
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM piezas WHERE id = :id"), {"id": id})
            conn.commit()
        return jsonify({"status": "deleted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/clasificar", methods=["POST"])
def clasificar():
    data = request.get_json()
    if not data or "imagen" not in data:
        return jsonify({"error": "No hay imagen"}), 400

    try:
        img_b64 = data["imagen"].split(",")[1] if "," in data["imagen"] else data["imagen"]
        img_bytes = base64.b64decode(img_b64)

        descriptor_app = calcular_hog(img_bytes)
        if descriptor_app is None:
            return jsonify({"error": "No se pudo procesar la imagen"}), 400

        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM piezas WHERE hog_descriptor IS NOT NULL"))
            piezas_db = result.mappings().all()

        mejor_match = None
        mejor_distancia = float('inf')

        for pieza in piezas_db:
            descriptor_db = texto_a_descriptor(pieza['hog_descriptor'])
            if descriptor_db is None:
                continue
            distancia = float(cosine(descriptor_app, descriptor_db))
            if distancia < mejor_distancia:
                mejor_distancia = distancia
                mejor_match = pieza

        if mejor_match and mejor_distancia < HOG_THRESHOLD:
            print(f"✅ Match: '{mejor_match['nombre']}' | Distancia HOG: {mejor_distancia:.4f}")
            return jsonify({
                "nombre":    mejor_match['nombre'],
                "cultura":   mejor_match['cultura'],
                "epoca":     mejor_match['epoca'],
                "material":  mejor_match['material'],
                "ubicacion": mejor_match['ubicacion'],
                "resumen":   mejor_match['resumen']
            })

        print(f"❌ Sin match | Mejor distancia: {mejor_distancia:.4f} | Umbral: {HOG_THRESHOLD}")
        return jsonify({"error": "Pieza no reconocida"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
