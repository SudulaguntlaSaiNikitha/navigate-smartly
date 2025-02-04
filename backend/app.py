from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from gtts import gTTS
import os
import tempfile
import threading
from datetime import datetime, time
import uuid
import cv2
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import base64
import io

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Create temporary directories
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# Load ML models
device = "cuda" if torch.cuda.is_available() else "cpu"

# Currency Detection Model
class CurrencyDetectionModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CurrencyDetectionModel, self).__init__()
        self.model = models.resnet34(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

try:
    currency_model = CurrencyDetectionModel().to(device)
    currency_model.load_state_dict(torch.load('best_weights/IC_ResNet34_9880.pth', map_location=device))
    currency_model.eval()
except Exception as e:
    print(f"Error loading currency model: {e}")

# Load other models
try:
    person_detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
    person_detection_model.eval()
except Exception as e:
    print(f"Error loading person detection model: {e}")

def cleanup_old_files():
    while True:
        current_time = datetime.now()
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            file_creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
            if (current_time - file_creation_time).seconds > 300:
                try:
                    os.remove(file_path)
                except:
                    pass
        time.sleep(300)

cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

def estimate_distance(box_height, frame_height):
    relative_size = box_height / frame_height
    if relative_size > 0.5:
        return "very close"
    elif relative_size > 0.3:
        return "close"
    elif relative_size > 0.15:
        return "medium distance"
    return "far"

@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    try:
        # Get base64 image from request
        data = request.get_json()
        base64_image = data['frame'].split(',')[1]
        image_data = base64.b64decode(base64_image)
        
        # Convert to PIL Image for currency detection
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image for currency detection
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Perform currency detection
        with torch.no_grad():
            outputs = currency_model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            
        # Map class index to currency value
        currency_values = {
            0: "10", 1: "20", 2: "50", 3: "100",
            4: "200", 5: "500", 6: "2000"
        }
        
        detected_value = currency_values.get(predicted.item(), None)
        
        # Convert to numpy array for person detection
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to tensor for person detection
        transform = transforms.ToTensor()
        img_tensor = transform(img).unsqueeze(0)

        # Detect persons
        with torch.no_grad():
            predictions = person_detection_model(img_tensor)[0]

        # Process person detection results
        boxes = predictions['boxes'].numpy()
        labels = predictions['labels'].numpy()
        scores = predictions['scores'].numpy()

        # Get person detections with distances
        persons = []
        for box, label, score in zip(boxes, labels, scores):
            if label == 1 and score > 0.6:  # Person class
                box_height = box[3] - box[1]
                distance = estimate_distance(box_height, img.shape[0])
                persons.append({
                    "distance": distance,
                    "confidence": float(score)
                })

        return jsonify({
            'persons': persons,
            'currency_value': detected_value,
            'frame_height': img.shape[0],
            'frame_width': img.shape[1]
        })

    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/speak', methods=['POST'])
def speak():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = app.make_default_options_response()
        return response
        
    try:
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', 'en')

        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(TEMP_DIR, filename)

        tts = gTTS(text=text, lang=language)
        tts.save(filepath)

        return send_file(
            filepath,
            mimetype='audio/mpeg',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        print(f"Error in /speak endpoint: {str(e)}")
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
