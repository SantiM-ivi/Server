from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import json
import os
import base64
import io
from PIL import Image
import imagehash  # Recordá poner ImageHash en requirements.txt
from datetime import datetime

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN ---
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 
UPLOAD_PARENT_DIR = 'biblioteca_museos'
DATABASE_NAME = 'museo.db'

if not os.path.exists(UPLOAD_PARENT_DIR):
    os.makedirs(UPLOAD_PARENT_DIR)

# --- SISTEMA DE BASE DE DATOS (AUTO-CREACIÓN) ---

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS piezas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            museo_id TEXT,
            nombre TEXT UNIQUE,
            cultura TEXT,
            epoca TEXT,
            material TEXT,
            ubicacion TEXT,
            resumen TEXT,
            hash_visual TEXT, 
            carpeta_fotos TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- FUNCIONES DE APOYO ---

def calcular_hash(imagen_bytes):
    """ Convierte una imagen en una huella digital matemática """
    try:
        img = Image.open(io.BytesIO(imagen_bytes))
        return str(imagehash.phash(img))
    except:
        return None

# --- RUTA PARA MANTENER EL SERVER DESPIERTO ---

@app.route('/')
def home():
    return "Servidor Activo 24/7 🚀", 200

# --- ENDPOINTS PARA EL PANEL WEB ---

@app.route("/web/subir_completo", methods=["POST"])
def subir_pieza_museo():
    try:
        meta = json.loads(request.form.get('metadata'))
        fotos = request.files.getlist('fotos')
        
        carpeta_pieza = os.path.join(UPLOAD_PARENT_DIR, meta['nombre'].replace(" ", "_"))
        os.makedirs(carpeta_pieza, exist_ok=True)
        
        hash_referencia = ""
        for i, foto in enumerate(fotos):
            contenido = foto.read()
            # Guardamos el hash de la primera foto como patrón
            if i == 0:
                hash_referencia = calcular_hash(contenido)
            
            with open(os.path.join(carpeta_pieza, f"angulo_{i+1}.jpg"), "wb") as f:
                f.write(contenido)
            
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO piezas 
            (museo_id, nombre, cultura, epoca, material, ubicacion, resumen, hash_visual, carpeta_fotos)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (meta['museo'], meta['nombre'], meta['cultura'], meta['epoca'], 
              meta['material'], meta['ubicacion'], meta['resumen'], hash_referencia, carpeta_pieza))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/web/lista", methods=["GET"])
def listar_piezas():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, nombre, museo_id FROM piezas ORDER BY id DESC")
    piezas = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(piezas)

@app.route("/web/obtener/<int:id>", methods=["GET"])
def obtener_pieza(id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM piezas WHERE id = ?", (id,))
    pieza = cursor.fetchone()
    conn.close()
    return jsonify(dict(pieza)) if pieza else ({}, 404)

@app.route("/web/editar/<int:id>", methods=["POST"])
def editar_pieza(id):
    try:
        meta = json.loads(request.form.get('metadata'))
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE piezas SET 
            museo_id=?, nombre=?, cultura=?, epoca=?, material=?, ubicacion=?, resumen=?
            WHERE id=?
        ''', (meta['museo'], meta['nombre'], meta['cultura'], meta['epoca'], 
              meta['material'], meta['ubicacion'], meta['resumen'], id))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- ENDPOINT DE CLASIFICACIÓN (APP GODOT) ---

@app.route("/clasificar", methods=["POST"])
def clasificar():
    data = request.get_json()
    if not data or "imagen" not in data:
        return jsonify({"error": "No hay imagen"}), 400

    try:
        # 1. Decodificar imagen de la App
        img_b64 = data["imagen"].split(",")[1] if "," in data["imagen"] else data["imagen"]
        img_bytes = base64.b64decode(img_b64)
        hash_app = imagehash.phash(Image.open(io.BytesIO(img_bytes)))
        
        # 2. Comparar visualmente con la Base de Datos
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM piezas")
        piezas_db = cursor.fetchall()
        conn.close()

        mejor_match = None
        umbral_similitud = 14 # Si la diferencia es menor a 14, se considera la misma pieza

        for pieza in piezas_db:
            if pieza['hash_visual']:
                hash_db = imagehash.hex_to_hash(pieza['hash_visual'])
                distancia = hash_app - hash_db
                
                if distancia < umbral_similitud:
                    umbral_similitud = distancia
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

        return jsonify({"error": "Pieza no reconocida en la base de datos"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))





