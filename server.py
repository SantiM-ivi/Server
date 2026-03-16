from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import re
import base64
import os
import sqlite3
import json
from datetime import datetime

app = Flask(__name__)
CORS(app) # Permite que el panel HTML se comunique con el servidor

# --- CONFIGURACIÓN ---
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 # Subidas de hasta 32MB
UPLOAD_PARENT_DIR = 'biblioteca_museos'
DATABASE_NAME = 'museo.db'

# Asegurar que existan las carpetas
if not os.path.exists(UPLOAD_PARENT_DIR):
    os.makedirs(UPLOAD_PARENT_DIR)

# --- CONFIGURACIÓN DE LLAVES ---
IMGBB_API_KEY = "89210d3875e24f75585ba5e2032b4566"
SERPAPI_KEY = "45fface95679af33c4823b73f9c49b5e0e6ef7514abfef8d162a5fb05174dae5"
EUROPEANA_API_KEY = "rierighobje"

# --- SISTEMA DE BASE DE DATOS (AUTO-CREACIÓN) ---

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Esta tabla se crea sola si no existe el archivo .db
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
            carpeta_fotos TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db() # Se ejecuta al encender el servidor

# --- LÓGICA DE PROCESAMIENTO ORIGINAL (TRADUCCIÓN Y FILTROS) ---

def traducir_texto(texto):
    if not texto or len(texto) < 3: return "En estudio"
    try:
        url = f"https://api.mymemory.translated.net/get?q={texto[:500]}&langpair=en|es"
        res = requests.get(url, timeout=5).json()
        return res.get('responseData', {}).get('translatedText', texto)
    except: return texto

# --- ENDPOINTS PARA EL PANEL WEB (CARGA Y EDICIÓN) ---

@app.route("/web/subir_completo", methods=["POST"])
def subir_pieza_museo():
    try:
        meta = json.loads(request.form.get('metadata'))
        fotos = request.files.getlist('fotos')
        
        # Crear carpeta para las 5 fotos
        carpeta_pieza = os.path.join(UPLOAD_PARENT_DIR, meta['nombre'].replace(" ", "_"))
        os.makedirs(carpeta_pieza, exist_ok=True)
        
        for i, foto in enumerate(fotos):
            foto.save(os.path.join(carpeta_pieza, f"angulo_{i+1}.jpg"))
            
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO piezas (museo_id, nombre, cultura, epoca, material, ubicacion, resumen, carpeta_fotos)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (meta['museo'], meta['nombre'], meta['cultura'], meta['epoca'], 
              meta['material'], meta['ubicacion'], meta['resumen'], carpeta_pieza))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"}), 201
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

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
        return jsonify({"status": "error", "message": str(e)}), 500

# --- ENDPOINT PRINCIPAL (USADO POR LA APP DE GODOT) ---

@app.route("/clasificar", methods=["POST"])
def clasificar():
    data = request.get_json()
    if not data or "imagen" not in data:
        return jsonify({"error": "No hay imagen"}), 400

    try:
        # 1. Identificación vía Google Lens (Tu lógica actual)
        img_b64 = data["imagen"].split(",")[1] if "," in data["imagen"] else data["imagen"]
        res_imgbb = requests.post("https://api.imgbb.com/1/upload", {"key": IMGBB_API_KEY}, files={"image": ("image.jpg", base64.b64decode(img_b64))}).json()
        image_url = res_imgbb["data"]["url"]

        lens_params = {"engine": "google_lens", "url": image_url, "api_key": SERPAPI_KEY, "hl": "es"}
        lens_data = requests.get("https://serpapi.com/search", params=lens_params).json()
        
        # Obtenemos el nombre que detectó Google
        nombre_detectado = ""
        if lens_data.get("knowledge_graph"):
            nombre_detectado = lens_data["knowledge_graph"][0].get("title", "")
        elif lens_data.get("visual_matches"):
            nombre_detectado = lens_data["visual_matches"][0].get("title", "")

        # 2. BUSCAR EN LA BASE DE DATOS DEL MUSEO (Prioridad)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM piezas WHERE nombre LIKE ?", (f'%{nombre_detectado}%',))
        pieza_local = cursor.fetchone()
        conn.close()

        if pieza_local:
            # Si el museo cargó la pieza, devolvemos su info oficial
            return jsonify({
                "nombre": pieza_local['nombre'],
                "cultura": pieza_local['cultura'],
                "epoca": pieza_local['epoca'],
                "material": pieza_local['material'],
                "ubicacion": pieza_local['ubicacion'],
                "resumen": pieza_local['resumen']
            })

        # 3. Si no está en la BD, podrías devolver error o buscar en Wikipedia (tu lógica vieja)
        return jsonify({"error": "Pieza no encontrada en los museos afiliados"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "Servidor de Museos Activo 🚀", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))







