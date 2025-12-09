import os
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = "models/skin_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["normal", "oily", "dry"]


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]

        if file.filename == "":
            return "No selected file"

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        img = preprocess_image(filepath)
        preds = model.predict(img)
        predicted_label = class_names[np.argmax(preds)]

        return render_template(
            "index.html",
            preview_image=file.filename,
            prediction_result=predicted_label
        )
    return render_template(
        "index.html",
        preview_image=None,
        prediction_result=None
    )


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
