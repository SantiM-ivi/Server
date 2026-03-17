from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import re
import base64
import os
import json
import io
from PIL import Image
import imagehash
from sqlalchemy import create_engine, text
from datetime import datetime

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN DE BASE DE DATOS (SUPABASE) ---
# Reemplaza 'TU_CONTRASEÑA_ACA' con tu clave real de Supabase
DB_URL = "postgresql://postgres:TU_CONTRASEÑA_ACA@db.oiijjpwfzgoprmjbsjrk.supabase.co:5432/postgres"
engine = create_engine(DB_URL)

# --- INICIALIZAR TABLA EN LA NUBE ---
def init_db():
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
                hash_visual TEXT,
                fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        '''))
        conn.commit()

init_db()

# --- FUNCIONES DE IA Y PROCESAMIENTO ---

def preparar_y_hash(imagen_bytes):
    try:
        img = Image.open(io.BytesIO(imagen_bytes)).convert('L')
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        return str(imagehash.phash(img))
    except:
        return None

def traducir_texto(texto):
    if not texto or len(texto) < 3: return "En estudio"
    try:
        url = f"https://api.mymemory.translated.net/get?q={texto[:500]}&langpair=en|es"
        res = requests.get(url, timeout=5).json()
        return res.get('responseData', {}).get('translatedText', texto)
    except: return texto

# --- RUTA DE SALUD ---
@app.route('/')
def home():
    return "Servidor con Base de Datos Permanente Activo 🚀", 200

# --- ENDPOINTS PANEL WEB ---

@app.route("/web/subir_completo", methods=["POST"])
def subir_pieza_museo():
    try:
        meta = json.loads(request.form.get('metadata'))
        fotos = request.files.getlist('fotos')
        
        # Generar hash de la primera foto (la principal de referencia)
        hash_ref = preparar_y_hash(fotos[0].read())
        
        with engine.connect() as conn:
            query = text('''
                INSERT INTO piezas (museo_id, nombre, cultura, epoca, material, ubicacion, resumen, hash_visual)
                VALUES (:m, :n, :c, :e, :mat, :u, :r, :h)
                ON CONFLICT (nombre) DO UPDATE SET
                cultura=EXCLUDED.cultura, epoca=EXCLUDED.epoca, material=EXCLUDED.material, 
                ubicacion=EXCLUDED.ubicacion, resumen=EXCLUDED.resumen, hash_visual=EXCLUDED.hash_visual
            ''')
            conn.execute(query, {
                "m": meta['museo'], "n": meta['nombre'], "c": meta['cultura'],
                "e": meta['epoca'], "mat": meta['material'], "u": meta['ubicacion'],
                "r": meta['resumen'], "h": hash_ref
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
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM piezas WHERE id = :id"), {"id": id})
        pieza = result.mappings().first()
    return jsonify(dict(pieza)) if pieza else ({}, 404)

@app.route("/web/editar/<int:id>", methods=["POST"])
def editar_pieza(id):
    try:
        meta = json.loads(request.form.get('metadata'))
        with engine.connect() as conn:
            conn.execute(text('''
                UPDATE piezas SET 
                museo_id=:m, nombre=:n, cultura=:c, epoca=:e, material=:mat, ubicacion=:u, resumen=:r
                WHERE id=:id
            '''), {
                "m": meta['museo'], "n": meta['nombre'], "c": meta['cultura'],
                "e": meta['epoca'], "mat": meta['material'], "u": meta['ubicacion'],
                "r": meta['resumen'], "id": id
            })
            conn.commit()
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- ENDPOINT CLASIFICAR (PARA LA APP DE GODOT) ---

@app.route("/clasificar", methods=["POST"])
def clasificar():
    data = request.get_json()
    if not data or "imagen" not in data:
        return jsonify({"error": "No hay imagen"}), 400

    try:
        # Decodificar imagen de Godot
        img_b64 = data["imagen"].split(",")[1] if "," in data["imagen"] else data["imagen"]
        img_bytes = base64.b64decode(img_b64)
        hash_app = imagehash.phash(Image.open(io.BytesIO(img_bytes)).convert('L'))
        
        # Traer todas las huellas digitales de la base de datos
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM piezas"))
            piezas_db = result.mappings().all()

        mejor_match = None
        menor_distancia = 26 # Umbral de tolerancia visual

        for pieza in piezas_db:
            if pieza['hash_visual']:
                hash_db = imagehash.hex_to_hash(pieza['hash_visual'])
                distancia = hash_app - hash_db
                
                if distancia < menor_distancia:
                    menor_distancia = distancia
                    mejor_match = pieza

        if mejor_match:
            return jsonify({
                "nombre": mejor_match['nombre'],
                "cultura": mejor_match['cultura'],
                "epoca": mejor_match['epoca'],
                "material": mejor_match['material'],
                "ubicacion": mejor_match['ubicacion'],
                "resumen": mejor_match['resumen']
            })

        return jsonify({"error": "Pieza no registrada"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))



