from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import json
import io
import numpy as np
from PIL import Image
import imagehash
from skimage.feature import hog
from skimage.transform import resize as sk_resize
from skimage.color import rgb2gray
from scipy.spatial.distance import cosine
from sqlalchemy import create_engine, text
from datetime import datetime

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN DE BASE DE DATOS (SUPABASE CON POOLER 6543) ---
DB_URL = "postgresql://postgres.oiijjpwfzgoprmjbsjrk:facugodot2026@aws-1-us-east-1.pooler.supabase.com:6543/postgres"

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    connect_args={'connect_timeout': 15}
)

def init_db():
    try:
        with engine.connect() as conn:
            # Tabla principal con columna para descriptor HOG
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
                    hash_visual TEXT,
                    hog_descriptor TEXT,
                    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''))
            # Migración: agregar la columna si la tabla ya existía sin ella
            conn.execute(text('''
                ALTER TABLE piezas
                ADD COLUMN IF NOT EXISTS hog_descriptor TEXT
            '''))
            conn.commit()
            print("✅ Base de Datos en Supabase lista y conectada.")
    except Exception as e:
        print(f"⚠️ Error de conexión inicial: {e}")

init_db()

# --- PARÁMETROS HOG ---
# Imagen normalizada a 128x128 para consistencia
HOG_IMAGE_SIZE = (128, 128)
HOG_PARAMS = {
    "orientations": 9,       # Número de bins de orientación
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": True,  # Mejora la invarianza a iluminación
}
# Umbral de similitud coseno: 0.0 = idéntico, 1.0 = opuesto
# Valores < 0.35 indican buena coincidencia para piezas arqueológicas
HOG_THRESHOLD = 0.35

# --- FUNCIONES DE APOYO ---

def calcular_hog(imagen_bytes: bytes) -> np.ndarray | None:
    """
    Recibe bytes de una imagen y devuelve el descriptor HOG como array numpy.
    Pasos:
      1. Abrir y convertir a escala de grises
      2. Redimensionar a tamaño canónico
      3. Calcular descriptor HOG
    """
    try:
        img = Image.open(io.BytesIO(imagen_bytes)).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_gray = rgb2gray(img_array)
        img_resized = sk_resize(img_gray, HOG_IMAGE_SIZE, anti_aliasing=True)
        descriptor = hog(img_resized, **HOG_PARAMS)
        return descriptor
    except Exception as e:
        print(f"⚠️ Error calculando HOG: {e}")
        return None

def descriptor_a_texto(descriptor: np.ndarray) -> str:
    """Serializa el array numpy a string JSON para guardar en la DB."""
    return json.dumps(descriptor.tolist())

def texto_a_descriptor(texto: str) -> np.ndarray | None:
    """Deserializa el descriptor desde la DB."""
    try:
        return np.array(json.loads(texto), dtype=np.float32)
    except:
        return None

def similitud_hog(desc1: np.ndarray, desc2: np.ndarray) -> float:
    """
    Distancia coseno entre dos descriptores HOG.
    Retorna valor entre 0.0 (idénticos) y 2.0 (opuestos).
    Menor es mejor.
    """
    return float(cosine(desc1, desc2))

def preparar_y_hash(imagen_bytes: bytes) -> str | None:
    """Hash perceptual pHash como referencia secundaria."""
    try:
        img = Image.open(io.BytesIO(imagen_bytes)).convert('L')
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        return str(imagehash.phash(img))
    except:
        return None

# --- RUTAS DEL SERVIDOR ---

@app.route('/')
def home():
    return "Servidor Permanente Activo (Supabase + HOG Visual ID) 🚀", 200

@app.route("/web/subir_completo", methods=["POST"])
def subir_pieza_museo():
    """
    Recibe metadata + foto de una pieza.
    Calcula hash pHash y descriptor HOG, guarda ambos en la DB.
    """
    try:
        meta = json.loads(request.form.get('metadata'))
        fotos = request.files.getlist('fotos')
        foto_bytes = fotos[0].read()

        hash_ref = preparar_y_hash(foto_bytes)
        descriptor = calcular_hog(foto_bytes)

        if descriptor is None:
            return jsonify({"error": "No se pudo calcular descriptor HOG de la imagen"}), 400

        hog_texto = descriptor_a_texto(descriptor)

        with engine.connect() as conn:
            query = text('''
                INSERT INTO piezas
                    (museo_id, nombre, cultura, epoca, material, ubicacion, resumen, hash_visual, hog_descriptor)
                VALUES
                    (:m, :n, :c, :e, :mat, :u, :r, :h, :hog)
                ON CONFLICT (nombre) DO UPDATE SET
                    cultura        = EXCLUDED.cultura,
                    epoca          = EXCLUDED.epoca,
                    material       = EXCLUDED.material,
                    ubicacion      = EXCLUDED.ubicacion,
                    resumen        = EXCLUDED.resumen,
                    hash_visual    = EXCLUDED.hash_visual,
                    hog_descriptor = EXCLUDED.hog_descriptor
            ''')
            conn.execute(query, {
                "m": meta['museo'], "n": meta['nombre'], "c": meta['cultura'],
                "e": meta['epoca'], "mat": meta['material'], "u": meta['ubicacion'],
                "r": meta['resumen'], "h": hash_ref, "hog": hog_texto
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
    """
    Recibe imagen en base64 desde la app Godot.

    Estrategia de matching en dos niveles:
      1. HOG  → distancia coseno (método principal, robusto a iluminación y escala)
      2. pHash → distancia Hamming (fallback si la pieza no tiene descriptor HOG aún)

    Retorna la pieza con menor distancia si supera el umbral configurado.
    """
    data = request.get_json()
    if not data or "imagen" not in data:
        return jsonify({"error": "No hay imagen"}), 400

    try:
        img_b64 = data["imagen"].split(",")[1] if "," in data["imagen"] else data["imagen"]
        img_bytes = base64.b64decode(img_b64)

        # Calcular HOG de la imagen entrante
        descriptor_app = calcular_hog(img_bytes)

        # Calcular pHash de la imagen entrante (fallback)
        hash_app = imagehash.phash(
            Image.open(io.BytesIO(img_bytes)).convert('L')
        )

        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM piezas"))
            piezas_db = result.mappings().all()

        mejor_match = None
        mejor_distancia = float('inf')
        metodo_usado = "ninguno"

        for pieza in piezas_db:

            # --- NIVEL 1: HOG (preferido) ---
            if descriptor_app is not None and pieza.get('hog_descriptor'):
                descriptor_db = texto_a_descriptor(pieza['hog_descriptor'])
                if descriptor_db is not None:
                    distancia = similitud_hog(descriptor_app, descriptor_db)
                    if distancia < mejor_distancia:
                        mejor_distancia = distancia
                        mejor_match = pieza
                        metodo_usado = "hog"
                    continue  # Si hay HOG, no usar pHash para esta pieza

            # --- NIVEL 2: pHash (fallback para piezas sin HOG) ---
            if pieza.get('hash_visual'):
                hash_db = imagehash.hex_to_hash(pieza['hash_visual'])
                # Normalizar distancia Hamming al rango [0, 1] para comparar con HOG
                distancia_normalizada = (hash_app - hash_db) / 64.0
                if distancia_normalizada < mejor_distancia:
                    mejor_distancia = distancia_normalizada
                    mejor_match = pieza
                    metodo_usado = "phash"

        # Aplicar umbral según método usado
        umbral_efectivo = HOG_THRESHOLD if metodo_usado == "hog" else 0.40

        if mejor_match and mejor_distancia < umbral_efectivo:
            print(f"✅ Match: '{mejor_match['nombre']}' | Método: {metodo_usado} | Distancia: {mejor_distancia:.4f}")
            return jsonify({
                "nombre":    mejor_match['nombre'],
                "cultura":   mejor_match['cultura'],
                "epoca":     mejor_match['epoca'],
                "material":  mejor_match['material'],
                "ubicacion": mejor_match['ubicacion'],
                "resumen":   mejor_match['resumen']
            })

        print(f"❌ Sin match | Mejor distancia: {mejor_distancia:.4f} | Umbral: {umbral_efectivo:.2f}")
        return jsonify({"error": "Pieza no reconocida"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
