from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import json
import os
import base64
import io
from PIL import Image
import imagehash  # Librería para comparar imágenes visualmente
from datetime import datetime

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN ---
DATABASE_NAME = 'museo.db'
UPLOAD_PARENT_DIR = 'biblioteca_museos'
if not os.path.exists(UPLOAD_PARENT_DIR):
    os.makedirs(UPLOAD_PARENT_DIR)

# --- BASE DE DATOS ---

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Agregamos 'hash_visual' para guardar la huella digital de la imagen
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

# --- FUNCIONES DE IA (HASHING VISUAL) ---

def calcular_hash(imagen_bytes):
    """ Convierte una imagen en una huella digital matemática """
    img = Image.open(io.BytesIO(imagen_bytes))
    # Usamos phash (Perceptual Hash) que es resistente a cambios de tamaño o luz
    return str(imagehash.phash(img))

# --- ENDPOINTS PARA EL PANEL WEB ---

@app.route("/web/subir_completo", methods=["POST"])
def subir_pieza_museo():
    try:
        meta = json.loads(request.form.get('metadata'))
        fotos = request.files.getlist('fotos')
        
        if len(fotos) < 5:
            return jsonify({"error": "Se requieren 5 fotos"}), 400

        # Guardar archivos físicamente
        carpeta_pieza = os.path.join(UPLOAD_PARENT_DIR, meta['nombre'].replace(" ", "_"))
        os.makedirs(carpeta_pieza, exist_ok=True)
        
        hashes = []
        for i, foto in enumerate(fotos):
            contenido = foto.read()
            # Guardar foto
            with open(os.path.join(carpeta_pieza, f"angulo_{i+1}.jpg"), "wb") as f:
                f.write(contenido)
            # Calcular hash de cada foto subida
            hashes.append(calcular_hash(contenido))
            
        # Guardamos el primer hash como referencia principal (o podrías promediarlos)
        hash_principal = hashes[0]

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO piezas 
            (museo_id, nombre, cultura, epoca, material, ubicacion, resumen, hash_visual, carpeta_fotos)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (meta['museo'], meta['nombre'], meta['cultura'], meta['epoca'], 
              meta['material'], meta['ubicacion'], meta['resumen'], hash_principal, carpeta_pieza))
        conn.commit()
        conn.close()
        
        print(f"✅ Pieza {meta['nombre']} guardada con hash: {hash_principal}")
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

# --- ENDPOINT DE CLASIFICACIÓN (APP GODOT) ---

@app.route("/clasificar", methods=["POST"])
def clasificar():
    data = request.get_json()
    if not data or "imagen" not in data:
        return jsonify({"error": "No hay imagen"}), 400

    try:
        # 1. Decodificar la imagen que manda la App
        img_b64 = data["imagen"].split(",")[1] if "," in data["imagen"] else data["imagen"]
        img_bytes = base64.b64decode(img_b64)
        
        # 2. Calcular el hash de la foto actual
        hash_actual = imagehash.phash(Image.open(io.BytesIO(img_bytes)))
        
        # 3. Comparar con TODA la base de datos
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM piezas")
        todas = cursor.fetchall()
        conn.close()

        mejor_match = None
        menor_distancia = 15 # Umbral de diferencia (0 es idéntico, >20 es muy diferente)

        for pieza in todas:
            if pieza['hash_visual']:
                hash_db = imagehash.hex_to_hash(pieza['hash_visual'])
                # Restar hashes da la "distancia". Si es pequeña, son la misma imagen.
                distancia = hash_actual - hash_db
                
                if distancia < menor_distancia:
                    menor_distancia = distancia
                    mejor_match = pieza

        if mejor_match:
            print(f"🎯 Match encontrado: {mejor_match['nombre']} (Distancia: {menor_distancia})")
            return jsonify({
                "nombre": mejor_match['nombre'],
                "cultura": mejor_match['cultura'],
                "epoca": mejor_match['epoca'],
                "material": mejor_match['material'],
                "ubicacion": mejor_match['ubicacion'],
                "resumen": mejor_match['resumen']
            })

        return jsonify({"error": "No se reconoció la pieza en la base de datos visual"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))






