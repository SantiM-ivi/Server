from flask import Flask, request, jsonify
from flask_cors import CORS
import base64, os, json, io, uuid, numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from sqlalchemy import create_engine, text
import requests as http_requests
import traceback

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN ---
DB_URL = "postgresql://postgres.oiijjpwfzgoprmjbsjrk:facugodot2026@aws-1-us-east-1.pooler.supabase.com:6543/postgres"
SIMILARITY_THRESHOLD = 0.5 # Ajustado para Normalización L2

engine = create_engine(DB_URL, pool_pre_ping=True)

class ArtAI:
    def __init__(self):
        print("📥 Cargando YOLO y MobileNet...")
        self.detector = YOLO('yolov8n.pt')
        self.extractor = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.extractor.classifier = nn.Identity()
        self.extractor.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_embedding(self, pil_img):
        img_t = self.preprocess(pil_img).unsqueeze(0)
        with torch.no_grad():
            emb = self.extractor(img_t)
        vec = emb.numpy().reshape(-1).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0: vec = vec / norm
        return vec.tolist()

    def smart_crop(self, pil_img):
        results = self.detector(pil_img, conf=0.15, verbose=False)
        w, h = pil_img.size
        if len(results[0].boxes) > 0:
            valid_boxes = []
            for b in results[0].boxes:
                box = b.xyxy[0].cpu().numpy()
                if ((box[2]-box[0]) * (box[3]-box[1])) > (w * h * 0.02):
                    valid_boxes.append(box)
            if valid_boxes:
                b = valid_boxes[0]
                return pil_img.crop((b[0], b[1], b[2], b[3]))
        return pil_img.crop((w*0.1, h*0.1, w*0.9, h*0.9))

ai = ArtAI()

@app.route('/')
def home(): return "Cerebro Activo 🧠", 200

# --- ESCÁNER (PARA GODOT) ---
@app.route('/clasificar', methods=['POST'])
def clasificar():
    try:
        data = request.get_json()
        if not data or "imagen" not in data:
            return jsonify({"error": "No se recibió imagen"}), 400
            
        img_data = data["imagen"]
        if "," in img_data: img_data = img_data.split(",")[1]
        
        img_bytes = base64.b64decode(img_data)
        raw_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # 1. IA: Crop y Embedding
        clean_img = ai.smart_crop(raw_img)
        vec = ai.get_embedding(clean_img)
        
        # 2. Supabase: Búsqueda Vectorial
        # IMPORTANTE: Convertimos el vector a string para que pgvector lo entienda
        vec_str = "[" + ",".join(map(str, vec)) + "]"
        
        with engine.connect() as conn:
            query = text("""
                SELECT nombre, cultura, epoca, material, ubicacion, resumen, 
                       (embedding <=> :v::vector) as distance
                FROM piezas 
                WHERE embedding IS NOT NULL 
                ORDER BY embedding <=> :v::vector 
                LIMIT 1
            """)
            res = conn.execute(query, {"v": vec_str}).mappings().first()
            
        if res:
            dist = float(res['distance'])
            if dist <= SIMILARITY_THRESHOLD:
                return jsonify({
                    "match": True,
                    "nombre": res['nombre'],
                    "cultura": res['cultura'],
                    "epoca": res['epoca'],
                    "material": res['material'],
                    "ubicacion": res['ubicacion'],
                    "resumen": res['resumen'],
                    "confianza": round((1 - dist) * 100, 2)
                }), 200
        
        return jsonify({"match": False, "distancia": float(res['distance']) if res else 2.0}), 404

    except Exception as e:
        # Esto te va a decir el error exacto en el log de Render
        print("❌ ERROR CRÍTICO:")
        print(traceback.format_exc())
        return jsonify({"error": str(e), "trace": "Ver logs del servidor"}), 500

# --- SUBIDA (PARA WEB) ---
@app.route('/web/subir_completo', methods=['POST'])
def subir():
    try:
        meta = json.loads(request.form.get('metadata'))
        fotos = request.files.getlist('fotos')
        
        # Embedding de la foto principal
        img_pil = Image.open(fotos[0]).convert('RGB')
        emb = ai.get_embedding(img_pil)
        vec_str = "[" + ",".join(map(str, emb)) + "]"

        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO piezas (museo_id, nombre, cultura, epoca, material, ubicacion, resumen, embedding)
                VALUES (:m, :n, :c, :e, :mat, :u, :r, :emb::vector)
                ON CONFLICT (nombre) DO UPDATE SET embedding = EXCLUDED.embedding
            """), {
                "m": meta['museo'], "n": meta['nombre'], "c": meta['cultura'], 
                "e": meta['epoca'], "mat": meta['material'], "u": meta['ubicacion'], 
                "r": meta['resumen'], "emb": vec_str
            })
            conn.commit()
        return jsonify({"status": "success"}), 201
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/web/lista")
def lista():
    with engine.connect() as conn:
        res = conn.execute(text("SELECT id, nombre, museo_id FROM piezas")).mappings().all()
        return jsonify([dict(r) for r in res])

@app.route("/web/borrar/<int:id>", methods=["DELETE"])
def borrar(id):
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM piezas WHERE id=:id"), {"id": id})
        conn.commit()
    return jsonify({"status": "deleted"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
