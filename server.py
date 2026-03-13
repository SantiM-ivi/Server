from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
import re
import base64
import io
import os

app = Flask(__name__)
# Permitir imágenes de hasta 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- CONFIGURACIÓN DE LLAVES ---
IMGBB_API_KEY = "89210d3875e24f75585ba5e2032b4566"
SERPAPI_KEY = "45fface95679af33c4823b73f9c49b5e0e6ef7514abfef8d162a5fb05174dae5"
EUROPEANA_API_KEY = "rierighobje" 

# --- RUTA PARA MANTENER EL SERVER DESPIERTO (UPTIMEROBOT) ---

@app.route('/')
def home():
    # Esta ruta responde con 200 OK para que el monitor no marque error
    return "Servidor Activo 24/7 🚀", 200

# --- LÓGICA DE PROCESAMIENTO Y FILTRADO ---

def traducir_texto(texto):
    if not texto or len(texto) < 3: return "En estudio"
    try:
        url = f"https://api.mymemory.translated.net/get?q={texto[:500]}&langpair=en|es"
        res = requests.get(url, timeout=5).json()
        return res.get('responseData', {}).get('translatedText', texto)
    except: return texto

def validar_filtro_estricto(corpus, ficha_keys):
    corpus = corpus.lower()
    # Bloqueo de Personas
    biograficos = ['born', 'died', 'nacimiento', 'fallecimiento', 'biografía', 'biography', 'spouse', 'politician', 'president']
    if any(p in str(ficha_keys).lower() for p in biograficos): return False

    # Filtros Positivos (ES/EN)
    filtros = [
        'dinosaur', 'fossil', 'fósil', 'inca', 'aztec', 'azteca', 'maya', 'egypt', 'egipto', 
        'roman', 'romano', 'pre-columbian', 'precolombino', 'colonial', 'stone age', 'bronze age', 
        'iron age', 'stone', 'piedra', 'pottery', 'cerámica', 'gold', 'oro', 'silver', 'plata', 
        'bronze', 'bronce', 'wood', 'madera', 'bone', 'hueso', 'textile', 'textil', 'vessel', 
        'vasija', 'mask', 'máscara', 'sculpture', 'escultura', 'weapon', 'arma', 'tool', 
        'herramienta', 'jewelry', 'joya', 'idol', 'ídolo', 'disc', 'disco', 'cretaceous', 'cretácico'
    ]
    exclusiones = ['toy', 'juguete', 'poster', 'fanart', 'plastic', 'plástico', 'modern replica']
    
    tiene_clave = any(f in corpus for f in filtros)
    es_moderno = any(e in corpus for e in exclusiones)
    return tiene_clave and not es_moderno

def buscar_palabras_clave(texto, lista_keywords):
    for word in lista_keywords:
        if word.lower() in texto.lower(): return word.capitalize()
    return "En estudio"

def consulta_europeana(termino):
    try:
        url = "https://api.europeana.eu/record/v2/search.json"
        params = {"wskey": EUROPEANA_API_KEY, "query": termino, "rows": 1, "profile": "rich"}
        res = requests.get(url, params=params, timeout=5).json()
        if res.get('items'):
            item = res['items'][0]
            return {
                'cultura': item.get('dcCreator', ['Desconocida'])[0],
                'epoca': item.get('year', ['En estudio'])[0],
                'material': item.get('dcType', ['No especificado'])[0],
                'ubicacion': item.get('dataProvider', ['Colección técnica'])[0]
            }
    except: return None

def extraer_datos_profundos(url, nombre_sugerido):
    ficha = {
        'nombre': nombre_sugerido.upper(),
        'cultura': 'En estudio',
        'epoca': 'En estudio',
        'material': 'En estudio',
        'ubicacion': 'En estudio',
        'dimensiones': 'En estudio',
        'resumen': 'Sin descripción.'
    }
    try:
        header = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=header, timeout=5)
        soup = BeautifulSoup(res.content, 'html.parser')
        texto_completo = soup.get_text()
        
        infobox = soup.find('table', {'class': ['infobox', 'vcard', 'biota']})
        if infobox:
            for row in infobox.find_all('tr'):
                th, td = row.find('th'), row.find('td')
                if th and td:
                    lbl, val = th.text.strip().lower(), re.sub(r'\[\d+\]', '', td.get_text(separator=" ").strip())
                    if any(x in lbl for x in ['cultura', 'culture', 'civilización', 'species', 'especie']): ficha['cultura'] = val
                    elif any(x in lbl for x in ['temporal range', 'rango temporal', 'lived', 'época', 'periodo']): ficha['epoca'] = val
                    elif any(x in lbl for x in ['material', 'medium', 'tipo']): ficha['material'] = val
                    elif any(x in lbl for x in ['ubicación', 'location', 'museo']): ficha['ubicacion'] = val
                    elif any(x in lbl for x in ['dimensiones', 'dimensions', 'height', 'longitud']): ficha['dimensiones'] = val

        # Respaldos por texto si la tabla falla
        if ficha['cultura'] == "En estudio":
            ficha['cultura'] = buscar_palabras_clave(texto_completo, ['Inca', 'Azteca', 'Maya', 'Egipto', 'Romano', 'Moche'])
        if ficha['material'] == "En estudio":
            ficha['material'] = buscar_palabras_clave(texto_completo, ['Oro', 'Piedra', 'Cerámica', 'Bronce', 'Plata', 'Hueso'])
        if ficha['epoca'] == "En estudio":
            ficha['epoca'] = buscar_palabras_clave(texto_completo, ['Cretácico', 'Jurásico', 'Siglo', 'BC', 'AD'])

        # Respaldo Europeana
        if ficha['cultura'] == "En estudio":
            eur = consulta_europeana(nombre_sugerido)
            if eur:
                ficha['cultura'] = eur['cultura']
                ficha['epoca'] = eur['epoca']
                ficha['material'] = eur['material']
                ficha['ubicacion'] = eur['ubicacion']

        p = soup.find('p', {'class': False}) or soup.find('p')
        if p: ficha['resumen'] = traducir_texto(p.text.strip())
        ficha['valid_corpus'] = texto_completo

    except: pass
    return ficha

# --- ENDPOINT PRINCIPAL ---

@app.route("/clasificar", methods=["POST"])
def clasificar():
    data = request.get_json()
    if not data or "imagen" not in data:
        return jsonify({"error": "No hay imagen"}), 400

    try:
        # 1. Subir a ImgBB para obtener URL pública
        img_b64 = data["imagen"].split(",")[1] if "," in data["imagen"] else data["imagen"]
        res_imgbb = requests.post(
            "https://api.imgbb.com/1/upload", 
            {"key": IMGBB_API_KEY}, 
            files={"image": ("image.jpg", base64.b64decode(img_b64))}
        ).json()
        image_url = res_imgbb["data"]["url"]

        # 2. Google Lens Identificación
        lens_params = {"engine": "google_lens", "url": image_url, "api_key": SERPAPI_KEY, "hl": "es"}
        lens_data = requests.get("https://serpapi.com/search", params=lens_params).json()
        matches = lens_data.get("visual_matches", [])

        # 3. Filtrado y Extracción
        ficha_final = None
        for m in matches[:10]:
            if "wikipedia.org" in m.get("link", ""):
                url_test = m["link"]
                nombre_test = lens_data.get("knowledge_graph", [{}])[0].get("title", m.get("title", "Objeto"))
                ficha_test = extraer_datos_profundos(url_test, nombre_test)
                
                if validar_filtro_estricto(ficha_test.get('valid_corpus', ''), ficha_test.keys()):
                    ficha_final = ficha_test
                    break

        if ficha_final:
            ficha_final.pop('valid_corpus', None)
            return jsonify(ficha_final)
        else:
            return jsonify({"error": "No se detectó una pieza arqueológica válida bajo los filtros establecidos."}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))








