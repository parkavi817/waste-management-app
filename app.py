from transformers import AutoProcessor, AutoModelForImageClassification
from PIL import Image, UnidentifiedImageError
import torch
import io
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_NAME = "google/vit-large-patch16-224-in21k"  
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

@app.route("/predict", methods=["POST"])
def predict():
    
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "No image file provided"}), 400
    
    try:
        
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    except UnidentifiedImageError:
        return jsonify({"error": "Invalid image file"}), 400
    
    
    inputs = processor(images=image, return_tensors="pt", padding=True, truncation=True)
    
    try:
       
        with torch.no_grad():
            logits = model(**inputs).logits
        
       
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]
        confidence = torch.softmax(logits, dim=-1).max().item()
        
        
        return jsonify({
            "prediction": predicted_class,
            "confidence": f"{confidence:.4f}"
        })
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
