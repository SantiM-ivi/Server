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

def calcular_hash(imagen_bytes):
    try:
        img = Image.open(io.BytesIO(imagen_bytes))
        # Convertimos a escala de grises y redimensionamos para que la comparación sea más estable
        return str(imagehash.phash(img))
    except:
        return None

@app.route('/')
def home():
    return "Servidor Activo 24/7 🚀", 200

# --- ENDPOINTS PANEL WEB ---

@app.route("/web/subir_completo", methods=["POST"])
def subir_pieza_museo():
    try:
        meta = json.loads(request.form.get('metadata'))
        fotos = request.files.getlist('fotos')
        
        carpeta_pieza = os.path.join(UPLOAD_PARENT_DIR, meta['nombre'].replace(" ", "_"))
        os.makedirs(carpeta_pieza, exist_ok=True)
        
        # Calculamos el hash de la PRIMERA foto cargada (la frontal)
        primer_foto = fotos[0].read()
        hash_ref = calcular_hash(primer_foto)
        
        # Guardar todas las fotos físicamente
        fotos[0].seek(0) # Resetear puntero para guardar
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
        print(f"✅ REGISTRADO: {meta['nombre']} con hash {hash_ref}")
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

# --- ENDPOINT DE CLASIFICACIÓN (EL QUE DA 404) ---

@app.route("/clasificar", methods=["POST"])
def clasificar():
    data = request.get_json()
    if not data or "imagen" not in data:
        return jsonify({"error": "No hay imagen"}), 400

    try:
        img_b64 = data["imagen"].split(",")[1] if "," in data["imagen"] else data["imagen"]
        img_bytes = base64.b64decode(img_b64)
        hash_app = imagehash.phash(Image.open(io.BytesIO(img_bytes)))
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM piezas")
        piezas_db = cursor.fetchall()
        conn.close()

        mejor_match = None
        # SUBIMOS EL UMBRAL: Antes era 14, ahora 24 para que sea MUCHO más fácil que coincida
        menor_distancia = 24 

        print(f"DEBUG: Hash de la App: {hash_app}")

        for pieza in piezas_db:
            if pieza['hash_visual']:
                hash_db = imagehash.hex_to_hash(pieza['hash_visual'])
                distancia = hash_app - hash_db
                
                print(f"DEBUG: Comparando con {pieza['nombre']} - Distancia: {distancia}")
                
                if distancia < menor_distancia:
                    menor_distancia = distancia
                    mejor_match = pieza

        if mejor_match:
            print(f"🎯 MATCH EXITOSO: {mejor_match['nombre']} (Distancia: {menor_distancia})")
            return jsonify({
                "nombre": mejor_match['nombre'],
                "cultura": mejor_match['cultura'],
                "epoca": mejor_match['epoca'],
                "material": mejor_match['material'],
                "ubicacion": mejor_match['ubicacion'],
                "resumen": mejor_match['resumen']
            })

        print("❌ ERROR: No hubo coincidencia cercana en la base de datos.")
        return jsonify({"error": "No reconocida"}), 404

    except Exception as e:
        print(f"⚠️ ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))




