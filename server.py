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
import gc

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN ---
DB_URL = "postgresql://postgres.oiijjpwfzgoprmjbsjrk:facugodot2026@aws-1-us-east-1.pooler.supabase.com:6543/postgres"
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_API_KEY = os.environ.get("SUPABASE_API_KEY", "")
STORAGE_BUCKET = "fotos-piezas"
SIMILARITY_THRESHOLD = 0.4  # Con Normalización L2, valores menores a 0.5 son muy seguros

engine = create_engine(DB_URL, pool_pre_ping=True)

# --- INICIALIZACIÓN DE MOTORES DE IA ---
class ArtAI:
    def __init__(self):
        # 1. Detector YOLOv8 (Versión Nano para no colapsar la RAM de Render)
        self.detector = YOLO('yolov8n.pt')
        
        # 2. Extractor de Embeddings (MobileNetV3 Small)
        self.extractor = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.extractor.classifier = nn.Identity()
        self.extractor.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def smart_crop(self, pil_img):
        """Lógica para evitar el 'problema del camello'"""
        w, h = pil_img.size
        # Guardamos temporalmente para YOLO
        temp_path = f"temp_{uuid.uuid4().hex}.jpg"
        pil_img.save(temp_path)
        
        results = self.detector(temp_path, conf=0.15, verbose=False)
        os.remove(temp_path) # Limpieza inmediata

        valid_boxes = []
        if len(results[0].boxes) > 0:
            for b in results[0].boxes:
                box = b.xyxy[0].cpu().numpy()
                area = (box[2]-box[0]) * (box[3]-box[1])
                # Filtro: El objeto debe ocupar al menos el 2% de la foto
                if area > (w * h * 0.02):
                    valid_boxes.append((box, area))
        
        if valid_boxes:
            valid_boxes.sort(key=lambda x: x[1], reverse=True)
            b = valid_boxes[0][0]
            return pil_img.crop((b[0], b[1], b[2], b[3]))
        
        # Fallback al centro si la IA falla
        return pil_img.crop((w*0.1, h*0.1, w*0.9, h*0.9))

    def get_embedding(self, pil_img):
        """Genera el vector normalizado L2"""
        img_t = self.preprocess(pil_img).unsqueeze(0)
        with torch.no_grad():
            emb = self.extractor(img_t)
        vec = emb.numpy().reshape(-1).astype(np.float32)
        # Normalización para Similitud de Coseno / Distancia L2 estable
        norm = np.linalg.norm(vec)
        if norm > 0: vec = vec / norm
        return vec.tolist()

ai = ArtAI()

# --- FUNCIONES DE ALMACENAMIENTO ---
def subir_foto_storage(imagen_bytes, pieza_nombre, indice):
    if not SUPABASE_URL or not SUPABASE_API_KEY: return None
    try:
        slug = pieza_nombre.lower().replace(' ', '_')[:30]
        path = f"{slug}/{indice}_{uuid.uuid4().hex[:6]}.jpg"
        url = f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{path}"
        headers = {
            "Authorization": f"Bearer {SUPABASE_API_KEY}",
            "Content-Type": "image/jpeg",
            "x-upsert": "true"
        }
        r = http_requests.post(url, headers=headers, data=imagen_bytes, timeout=15)
        if r.status_code in (200, 201):
            return f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{path}"
        return None
    except: return None

# --- RUTAS ---

@app.route('/')
def home():
    return "🚀 Servidor de Reconocimiento Arqueológico Activo", 200

# RUTA PARA GODOT (ESCÁNER)
@app.route('/clasificar', methods=['POST'])
def clasificar():
    try:
        data = request.get_json()
        img_data = data["imagen"]
        if "," in img_data: img_data = img_data.split(",")[1]
        
        img_bytes = base64.b64decode(img_data)
        raw_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # 1. Limpieza de imagen (Smart Crop)
        clean_img = ai.smart_crop(raw_img)
        
        # 2. Generar Vector
        vec = ai.get_embedding(clean_img)
        
        # 3. Búsqueda en Supabase con pgvector (<=> es distancia de coseno)
        with engine.connect() as conn:
            query = text("""
                SELECT nombre, cultura, epoca, material, ubicacion, resumen, fotos_urls[1] as foto,
                       (embedding <=> :v::vector) as dist
                FROM piezas 
                WHERE embedding IS NOT NULL 
                ORDER BY embedding <=> :v::vector 
                LIMIT 1
            """)
            res = conn.execute(query, {"v": str(vec)}).mappings().first()
            
        if res and res['dist'] <= SIMILARITY_THRESHOLD:
            return jsonify({
                "match": True,
                "nombre": res['nombre'],
                "cultura": res['cultura'],
                "epoca": res['epoca'],
                "material": res['material'],
                "ubicacion": res['ubicacion'],
                "resumen": res['resumen'],
                "foto_url": res['foto'],
                "confianza": round((1 - float(res['dist'])) * 100, 2)
            }), 200
            
        return jsonify({"match": False, "dist": float(res['dist']) if res else 2.0}), 404

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# RUTA PARA PANEL WEB (CARGA DE PIEZAS)
@app.route('/web/subir_completo', methods=['POST'])
def subir_pieza_museo():
    try:
        meta = json.loads(request.form.get('metadata'))
        fotos = request.files.getlist('fotos')
        if not fotos: return jsonify({"error": "Falta imagen"}), 400
        
        # Procesamos la primera foto para el embedding (foto de catálogo)
        primera_foto_bytes = fotos[0].read()
        img_pil = Image.open(io.BytesIO(primera_foto_bytes)).convert('RGB')
        
        # Generamos embedding (sin crop, se asume que el catálogo es limpio)
        emb = ai.get_embedding(img_pil)
        
        # Subir todas las fotos a Storage
        urls = []
        fotos[0].seek(0) # Reset para volver a leerla
        for i, f in enumerate(fotos):
            fb = f.read()
            u = subir_foto_storage(fb, meta['nombre'], i)
            if u: urls.append(u)
        
        with engine.connect() as conn:
            conn.execute(text('''
                INSERT INTO piezas (museo_id, nombre, cultura, epoca, material, ubicacion, resumen, embedding, fotos_urls)
                VALUES (:m, :n, :c, :e, :mat, :u, :r, :emb, :urls)
                ON CONFLICT (nombre) DO UPDATE SET 
                    embedding=EXCLUDED.embedding, fotos_urls=EXCLUDED.fotos_urls
            '''), {
                "m": meta['museo'], "n": meta['nombre'], "c": meta['cultura'], "e": meta['epoca'], 
                "mat": meta['material'], "u": meta['ubicacion'], "r": meta['resumen'], 
                "emb": str(emb), "urls": urls
            })
            conn.commit()
            
        gc.collect() # Liberar memoria
        return jsonify({"status": "success", "urls": urls}), 201
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# --- MANTENEMOS TUS OTRAS RUTAS (LISTA, BORRAR, ETC.) ---
@app.route("/web/lista", methods=["GET"])
def listar_piezas():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT id, nombre, museo_id FROM piezas ORDER BY id DESC"))
        return jsonify([dict(r) for r in result.mappings().all()])

@app.route("/web/borrar/<int:id>", methods=["DELETE"])
def borrar_pieza(id):
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM piezas WHERE id=:id"), {"id": id})
        conn.commit()
    return jsonify({"status": "deleted"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
