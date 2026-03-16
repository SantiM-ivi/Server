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
CORS(app)

# --- CONFIGURACIÓN ---
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
UPLOAD_PARENT_DIR = 'biblioteca_museos'
DATABASE_NAME = 'museo.db'

if not os.path.exists(UPLOAD_PARENT_DIR):
    os.makedirs(UPLOAD_PARENT_DIR)

IMGBB_API_KEY = "89210d3875e24f75585ba5e2032b4566"
SERPAPI_KEY = "45fface95679af33c4823b73f9c49b5e0e6ef7514abfef8d162a5fb05174dae5"

# --- BASE DE DATOS ---

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
            carpeta_fotos TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- LÓGICA DE PROCESAMIENTO ---

def traducir_texto(texto):
    if not texto or len(texto) < 3: return "En estudio"
    try:
        url = f"https://api.mymemory.translated.net/get?q={texto[:500]}&langpair=en|es"
        res = requests.get(url, timeout=5).json()
        return res.get('responseData', {}).get('translatedText', texto)
    except: return texto

def es_humano_real(texto_analizar):
    """
    Único filtro activo: Bloquea si la descripción contiene palabras 
    relacionadas con personas reales.
    """
    bloqueo = ['person', 'human', 'man', 'woman', 'selfie', 'persona', 'hombre', 'mujer', 'boy', 'girl', 'face', 'rostro']
    texto_analizar = texto_analizar.lower()
    return any(p in texto_analizar for p in bloqueo)

def extraer_datos_wikipedia(url, nombre_sugerido):
    ficha = {
        'nombre': nombre_sugerido.upper(),
        'cultura': 'Información general',
        'epoca': 'No especificada',
        'material': 'Varios',
        'ubicacion': 'Ver descripción',
        'resumen': 'Sin descripción disponible.'
    }
    try:
        header = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=header, timeout=5)
        soup = BeautifulSoup(res.content, 'html.parser')
        
        # Filtro de seguridad humano
        if es_humano_real(soup.get_text()):
            return None

        # Intentar sacar resumen del primer párrafo
        p = soup.find('p', {'class': False}) or soup.find('p')
        if p: ficha['resumen'] = traducir_texto(p.text.strip())
        
        # Intentar extraer datos de la tabla (infobox)
        infobox = soup.find('table', {'class': ['infobox']})
        if infobox:
            for row in infobox.find_all('tr'):
                th, td = row.find('th'), row.find('td')
                if th and td:
                    lbl, val = th.text.strip().lower(), td.text.strip()
                    if any(x in lbl for x in ['cultura', 'culture', 'civilización']): ficha['cultura'] = val
                    elif any(x in lbl for x in ['época', 'period', 'año']): ficha['epoca'] = val
                    elif any(x in lbl for x in ['material', 'medium']): ficha['material'] = val
                    elif any(x in lbl for x in ['ubicación', 'location', 'museo']): ficha['ubicacion'] = val
    except: pass
    return ficha

# --- ENDPOINTS WEB ---

@app.route("/web/subir_completo", methods=["POST"])
def subir_pieza_museo():
    try:
        meta = json.loads(request.form.get('metadata'))
        fotos = request.files.getlist('fotos')
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
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/web/lista", methods=["GET"])
def listar_piezas():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, nombre, museo_id FROM piezas ORDER BY id DESC")
    piezas = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(piezas)

# --- ENDPOINT CLASIFICAR (GODOT) ---

@app.route("/clasificar", methods=["POST"])
def clasificar():
    data = request.get_json()
    if not data or "imagen" not in data:
        return jsonify({"error": "No hay imagen"}), 400

    try:
        # 1. Subir y buscar en Google Lens
        img_b64 = data["imagen"].split(",")[1] if "," in data["imagen"] else data["imagen"]
        res_imgbb = requests.post("https://api.imgbb.com/1/upload", {"key": IMGBB_API_KEY}, 
                                  files={"image": ("image.jpg", base64.b64decode(img_b64))}).json()
        image_url = res_imgbb["data"]["url"]

        lens_params = {"engine": "google_lens", "url": image_url, "api_key": SERPAPI_KEY, "hl": "es"}
        lens_data = requests.get("https://serpapi.com/search", params=lens_params).json()
        
        nombre_detectado = ""
        if lens_data.get("knowledge_graph"):
            nombre_detectado = lens_data["knowledge_graph"][0].get("title", "")
        elif lens_data.get("visual_matches"):
            nombre_detectado = lens_data["visual_matches"][0].get("title", "")

        # 2. BUSCAR EN BD LOCAL (Lo que subieron los museos)
        if nombre_detectado:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM piezas WHERE nombre LIKE ?", (f'%{nombre_detectado}%',))
            pieza_local = cursor.fetchone()
            conn.close()
            if pieza_local:
                return jsonify(dict(pieza_local))

        # 3. SI NO ESTÁ EN BD, BUSCAR EN WIKIPEDIA (Sin filtros de tipo de objeto)
        for m in lens_data.get("visual_matches", [])[:8]:
            if "wikipedia.org" in m.get("link", ""):
                ficha = extraer_datos_wikipedia(m["link"], nombre_detectado or m["title"])
                if ficha: # Si no es un humano, devolver
                    return jsonify(ficha)

        # 4. ÚLTIMO RECURSO: Devolver lo que diga Google Lens directamente
        if nombre_detectado:
            return jsonify({
                "nombre": nombre_detectado.upper(),
                "cultura": "General",
                "epoca": "Desconocida",
                "material": "No especificado",
                "ubicacion": "Búsqueda web",
                "resumen": "Se detectó el objeto pero no hay ficha técnica detallada."
            })

        return jsonify({"error": "No se pudo identificar el objeto"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "Servidor Híbrido Activo 🚀", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))







