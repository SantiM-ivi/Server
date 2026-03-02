from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image as Image
import requests

app = Flask(__name__)

# Cargar modelo una sola vez
model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5")
labels_path = tf.keras.utils.get_file(
    "ImageNetLabels.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
)
labels = open(labels_path).read().splitlines()

# Endpoint raíz para probar en navegador
@app.route("/")
def home():
    return "Servidor online 🚀"

# Endpoint de clasificación
@app.route("/clasificar", methods=["POST"])
def clasificar():
    file = request.files["imagen"]
    img = Image.open(file).resize((224, 224))
    img = np.array(img) / 255.0
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    predictions = model(img)
    predicted_class = np.argmax(predictions[0])
    etiqueta = labels[predicted_class]

    # Consultar Met Museum
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

