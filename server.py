import base64
import io

@app.route("/clasificar", methods=["POST"])
def clasificar():
    data = request.get_json()
    
    if not data or "imagen" not in data:
        return jsonify({"error": "No se recibió el campo 'imagen' en el JSON"}), 400

    try:
        # Decodificar la imagen desde Base64
        image_data = base64.b64decode(data["imagen"])
        file = io.BytesIO(image_data)
        
        # El resto del código de procesamiento sigue igual...
        img = Image.open(file).resize((224, 224))
        # ... (tus predicciones de TensorFlow y búsqueda en el Met) ...
        
        # (Asegúrate de que el resto del código sea igual al que tenías)
        img = np.array(img) / 255.0
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        predictions = model(img)
        predicted_class = np.argmax(predictions[0])
        etiqueta = labels[predicted_class]
        
        # Consultar Met Museum...
        search_url = f"https://collectionapi.metmuseum.org/public/collection/v1/search?q={etiqueta}"
        response = requests.get(search_url).json()
        result = {"query": etiqueta}
        if response.get("total", 0) > 0:
            object_id = response["objectIDs"][0]
            object_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
            object_data = requests.get(object_url).json()
            result.update({
                "title": object_data.get("title", ""),
                "artist": object_data.get("artistDisplayName", ""),
                "date": object_data.get("objectDate", ""),
                "department": object_data.get("department", ""),
                "url": object_data.get("objectURL", "")
            })
        return jsonify(result)

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500




