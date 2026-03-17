from flask import Flask, request, jsonify
from flask_cors import CORS
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

# --- CONFIGURACIÓN DE BASE DE DATOS (SUPABASE CON POOLER 6543) ---
DB_URL = "postgresql://postgres.oiijjpwfzgoprmjbsjrk:facugodot2026@aws-1-us-east-1.pooler.supabase.com:6543/postgres"

# Configuramos el motor para ser resistente a la red de Render
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
                    hash_visual TEXT,
                    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''))
            conn.commit()
            print("✅ Base de Datos en Supabase lista y conectada.")
    except Exception as e:
        print(f"⚠️ Error de conexión inicial: {e}")

init_db()

# --- FUNCIONES DE APOYO ---

def preparar_y_hash(imagen_bytes):
    try:
        img = Image.open(io.BytesIO(imagen_bytes)).convert('L')
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        return str(imagehash.phash(img))
    except:
        return None

# --- RUTAS DEL SERVIDOR ---

@app.route('/')
def home():
    return "Servidor Permanente Activo (Supabase + Visual ID) 🚀", 200

@app.route("/web/subir_completo", methods=["POST"])
def subir_pieza_museo():
    try:
        meta = json.loads(request.form.get('metadata'))
        fotos = request.files.getlist('fotos')
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
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM piezas WHERE id = :id"), {"id": id})
            pieza = result.mappings().first()
        return jsonify(dict(pieza)) if pieza else ({}, 404)
    except:
        return jsonify({}), 500

# --- NUEVA RUTA: BORRAR PIEZA ---
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
        hash_app = imagehash.phash(Image.open(io.BytesIO(img_bytes)).convert('L'))
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM piezas"))
            piezas_db = result.mappings().all()

        mejor_match = None
        menor_distancia = 26 
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
        return jsonify({"error": "Pieza no reconocida"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

