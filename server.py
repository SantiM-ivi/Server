from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import json
import os
import base64
import io
from PIL import Image
import imagehash 
from datetime import datetime

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN ---
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 
UPLOAD_PARENT_DIR = 'biblioteca_museos'
DATABASE_NAME = 'museo.db'

if not os.path.exists(UPLOAD_PARENT_DIR):
    os.makedirs(UPLOAD_PARENT_DIR)

# --- SISTEMA DE BASE DE DATOS ---

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

# Inicializar DB al arrancar
init_db()

# --- MEJORA EN EL CÁLCULO DEL HASH (MÁS TOLERANTE) ---
def preparar_y_hash(imagen_bytes):
    try:
        # Convertir a escala de grises y redimensionar para que luz/color no afecten tanto
        img = Image.open(io.BytesIO(imagen_bytes)).convert('L')
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        return imagehash.phash(img)
    except Exception as e:
        print(f"Error procesando imagen: {e}")
        return None

# --- RUTA PARA MANTENER EL SERVER DESPIERTO (OBLIGATORIA) ---

@app.route('/')
def home():
    # No tocar esta ruta para que UptimeRobot funcione
    return "Servidor Activo 24/7 🚀", 200

# --- ENDPOINTS PANEL WEB ---

@app.route("/web/subir_completo", methods=["POST"])
def subir_pieza_museo():
    try:
        meta = json.loads(request.form.get('metadata'))
        fotos = request.files.getlist('fotos')
        
        carpeta_pieza = os.path.join(UPLOAD_PARENT_DIR, meta['nombre'].replace(" ", "_"))
        os.makedirs(carpeta_pieza, exist_ok=True)
        
        # Generar hash de referencia usando la primera foto (frontal)
        primer_foto_data = fotos[0].read()
        hash_ref = str(preparar_y_hash(primer_foto_data))
        
        # Guardar físicamente las 5 fotos
        fotos[0].seek(0)
        for i, foto in enumerate(fotos):
            foto.save(os.path.join(carpeta_pieza, f"angulo_{i+1}.jpg"))
            
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO piezas 
            (museo_id, nombre, cultura, epoca, material, ubicacion, resumen, hash_visual, carpeta_fotos)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (meta['museo'], meta['nombre'], meta['cultura'], meta['epoca'], 
              meta['material'], meta['ubicacion'], meta['resumen'], hash_ref, carpeta_pieza))
        conn.commit()
        conn.close()
        
        print(f"✅ REGISTRADO: {meta['nombre']} | Hash: {hash_ref}")
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

# --- ENDPOINT CLASIFICAR (GODOT) ---

@app.route("/clasificar", methods=["POST"])
def clasificar():
    data = request.get_json()
    if not data or "imagen" not in data:
        return jsonify({"error": "No hay imagen"}), 400

    try:
        # Decodificar imagen de la App
        img_b64 = data["imagen"].split(",")[1] if "," in data["imagen"] else data["imagen"]
        img_bytes = base64.b64decode(img_b64)
        hash_app = preparar_y_hash(img_bytes)
        
        if hash_app is None:
            return jsonify({"error": "Error al procesar imagen de la App"}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM piezas")
        piezas_db = cursor.fetchall()
        conn.close()

        if not piezas_db:
            print("⚠️ DB VACÍA")
            return jsonify({"error": "No hay datos cargados"}), 404

        mejor_match = None
        menor_distancia = 64
        umbral_sensibilidad = 28 # Más alto = más fácil reconocer (máximo 64)

        for pieza in piezas_db:
            if pieza['hash_visual']:
                hash_db = imagehash.hex_to_hash(pieza['hash_visual'])
                distancia = hash_app - hash_db
                
                print(f"Comparando con {pieza['nombre']} | Distancia: {distancia}")
                
                if distancia < menor_distancia:
                    menor_distancia = distancia
                    mejor_match = pieza

        if mejor_match and menor_distancia <= umbral_sensibilidad:
            print(f"🎯 MATCH: {mejor_match['nombre']} (D: {menor_distancia})")
            return jsonify({
                "nombre": mejor_match['nombre'],
                "cultura": mejor_match['cultura'],
                "epoca": mejor_match['epoca'],
                "material": mejor_match['material'],
                "ubicacion": mejor_match['ubicacion'],
                "resumen": mejor_match['resumen']
            })

        print(f"❌ FALLO: Distancia mínima fue {menor_distancia}")
        return jsonify({"error": "No reconocida"}), 404

    except Exception as e:
        print(f"⚠️ ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))



