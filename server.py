from flask import Flask, request, jsonify
from flask_cors import CORS
import base64, os, json, io, uuid, numpy as np
from PIL import Image
import onnxruntime as ort
from sqlalchemy import create_engine, text
import requests as http_requests

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
DB_URL = "postgresql://postgres.oiijjpwfzgoprmjbsjrk:facugodot2026@aws-1-us-east-1.pooler.supabase.com:6543/postgres"
# Buscamos el archivo ONNX en la raíz con diferentes nombres posibles
MODEL_PATH = "modelo_museo.onnx"
SIMILARITY_THRESHOLD = 0.75

engine = create_engine(DB_URL, pool_pre_ping=True)

# Intentar cargar la sesión de ONNX al arrancar
session = None
try:
    if MODEL_PATH:
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        model_status = f"✅ ONNX Cargado: {MODEL_PATH}"
    else:
        model_status = "❌ ARCHIVO .ONNX NO ENCONTRADO EN EL REPOSITORIO"
except Exception as e:
    model_status = f"❌ ERROR AL INICIALIZAR ONNX: {str(e)}"

# ---------------------------------------------------------------------------
# RUTA DE ESTADO (Para verificar en el navegador)
# ---------------------------------------------------------------------------
@app.route('/')
def home():
    archivos_en_root = os.listdir('.')
    return f"""
    <h1>🚀 Servidor Smartify Activo</h1>
    <p><b>Estado:</b> {model_status}</p>
    <p><b>Archivos detectados:</b> {archivos_en_root}</p>
    """, 200

# ---------------------------------------------------------------------------
# RUTA DE CLASIFICACIÓN (GODOT)
# ---------------------------------------------------------------------------
@app.route('/clasificar', methods=['POST'])
def clasificar():
    if not session:
        return jsonify({"error": "Modelo IA no cargado. Revisa la ruta principal del server."}), 503

    try:
        # 1. Validar JSON
        data = request.get_json(silent=True)
        if not data or "imagen" not in data:
            return jsonify({"error": "JSON invalido o falta el campo 'imagen'"}), 400

        # 2. Procesar Imagen
        try:
            img_b64 = data["imagen"].split(",")[1] if "," in data["imagen"] else data["imagen"]
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
        except Exception as e:
            return jsonify({"error": f"Error en decodificacion/procesamiento de imagen: {str(e)}"}), 400

        # 3. Inferencia de IA (Embedding)
        try:
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            arr = arr.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
            
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            embedding = session.run([output_name], {input_name: arr})[0][0]
            
            norm = np.linalg.norm(embedding)
            vec = (embedding / norm).tolist() if norm > 0 else None
        except Exception as e:
            return jsonify({"error": f"Error en la inferencia del modelo ONNX: {str(e)}"}), 500

        # 4. Busqueda Vectorial en Supabase
        try:
            with engine.connect() as conn:
                query = text("""
                    SELECT nombre, cultura, epoca, material, ubicacion, resumen, fotos_urls[1] as foto,
                    1 - (embedding <=> :v::vector) as sim
                    FROM piezas 
                    WHERE embedding IS NOT NULL 
                    ORDER BY embedding <=> :v::vector 
                    LIMIT 1
                """)
                res = conn.execute(query, {"v": str(vec)}).mappings().first()
        except Exception as e:
            return jsonify({"error": f"Error en la consulta a Base de Datos (pgvector): {str(e)}"}), 500

        # 5. Resultado
        if res and res['sim'] >= SIMILARITY_THRESHOLD:
            return jsonify({
                "match": True,
                "nombre": res['nombre'],
                "cultura": res['cultura'],
                "epoca": res['epoca'],
                "material": res['material'],
                "ubicacion": res['ubicacion'],
                "resumen": res['resumen'],
                "foto_url": res['foto'],
                "confianza": round(float(res['sim']) * 100, 2)
            }), 200
        
        return jsonify({
            "match": False, 
            "confianza": round(float(res['sim']) * 100, 2) if res else 0,
            "msg": "No se alcanzo el umbral de similitud"
        }), 404

    except Exception as e:
        return jsonify({"error": f"Excepcion Global: {str(e)}"}), 500

# ---------------------------------------------------------------------------
# RUTAS PANEL WEB (MANTENIDAS)
# ---------------------------------------------------------------------------
@app.route("/web/lista", methods=["GET"])
def listar_piezas():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT id, nombre, museo_id FROM piezas ORDER BY id DESC"))
            return jsonify([dict(r) for r in result.mappings().all()])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/web/borrar/<int:id>", methods=["DELETE"])
def borrar_pieza(id):
    try:
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM piezas WHERE id=:id"), {"id": id})
            conn.commit()
        return jsonify({"status": "deleted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
