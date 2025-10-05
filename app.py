import os
import time
import io
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.serialization
from torchvision import transforms
import torchvision.models as tvmodels 
from torchvision.models.resnet import ResNet
from PIL import Image
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask_bcrypt import Bcrypt
import requests
from flask import Flask, send_from_directory

def download_file(url, save_path):
    if not os.path.exists(save_path):
        print(f"Model file not found. Downloading from {url}...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(save_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Downloaded {os.path.basename(save_path)} successfully.")
        except Exception as e:
            # If download fails, delete the partially downloaded file
            if os.path.exists(save_path):
                os.remove(save_path)
            raise RuntimeError(f"Failed to download model file from {url}. Error: {e}")
# Allowlist the layers/activations your model uses
torch.serialization.add_safe_globals([
    nn.Conv2d,
    nn.Linear,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.MaxPool2d,
    nn.AdaptiveAvgPool2d,
    nn.Sequential
])
 

# ------------------------
# 1. Flask App Setup
# ------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
bcrypt = Bcrypt(app)

basedir = os.path.abspath(os.path.dirname(__file__))
database_url = os.environ.get('DATABASE_URL')
if database_url:
    # On Render, SQLAlchemy needs 'postgresql://' instead of 'postgres://'
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url.replace("postgres://", "postgresql://")
else:
    # For local testing, use the old SQLite database
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'skin_detector.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)

# --- Get Model URLs from Environment Variables and Download Models ---
PROTOTXT_URL = os.environ.get('PROTOTXT_URL')
CAFFE_MODEL_URL = os.environ.get('CAFFE_MODEL_URL')
PYTORCH_MODEL_URL = os.environ.get('PYTORCH_MODEL_URL')

TEMP_DIR = '/tmp'
prototxt_path = os.path.join(basedir, 'deploy.prototxt')
weights_path = os.path.join(basedir, 'res10_300x300_ssd_iter_140000.caffemodel')
MODEL_PATH = os.path.join('/tmp', 'skin_type_detector_app_full.pth')

print("Checking for model files...")
download_file(PROTOTXT_URL, prototxt_path)
download_file(CAFFE_MODEL_URL, weights_path)
download_file(PYTORCH_MODEL_URL, MODEL_PATH)

# --- Load models from the (now existing) local files ---
net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
print("✅ Caffe DNN face detector loaded successfully.")

print(f"DEBUG: prototxt size: {os.path.getsize(prototxt_path)} bytes, caffemodel size: {os.path.getsize(weights_path)} bytes")

# ------------------------
# 2. Database Models
# ------------------------
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    password_hash = db.Column(db.String(200), nullable=False)
    avatar = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow) 
    scans = db.relationship('Scan', backref='user', lazy=True)

class Scan(db.Model):
    __tablename__ = 'scan_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    skin_type = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# ------------------------
# 3. Load PyTorch Model (robust)
# ------------------------
MODEL = None
MODEL_PATH = os.path.join(basedir, 'skin_type_detector_app_full.pth')
SKIN_CLASSES = ['dry', 'normal', 'oily']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    global MODEL
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
     
    with torch.serialization.safe_globals([
    nn.Conv2d,
    nn.Linear,
    nn.BatchNorm2d,
    ResNet
    ]):
        checkpoint = torch.load(MODEL_PATH, map_location=device,weights_only=False)


    # If the checkpoint is already a module (saved with torch.save(model)), use it
    if isinstance(checkpoint, nn.Module):
        MODEL = checkpoint
    elif isinstance(checkpoint, dict):
        # If checkpoint stores whole model in a key
        if 'model' in checkpoint and isinstance(checkpoint['model'], nn.Module):
            MODEL = checkpoint['model']
        else:
            # Interpret checkpoint as a state_dict
            state_dict = checkpoint.get('state_dict', checkpoint)

            # Try a ResNet18-based architecture first (common)
            try:
                model = tvmodels.resnet50(weights=None)
                model.fc = nn.Linear(model.fc.in_features, len(SKIN_CLASSES))
                model.load_state_dict(state_dict)
                MODEL = model
            except Exception as e_resnet:
                # Try a small fallback CNN architecture
                try:
                    class SimpleCNN(nn.Module):
                        def __init__(self, num_classes=3):
                            super(SimpleCNN, self).__init__()
                            self.features = nn.Sequential(
                                nn.Conv2d(3, 16, 3, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                nn.Conv2d(16, 32, 3, padding=1),
                                nn.ReLU(),
                                nn.AdaptiveAvgPool2d((1,1)),
                                nn.Flatten()
                            )
                            self.classifier = nn.Linear(32, num_classes)

                        def forward(self, x):
                            x = self.features(x)
                            x = self.classifier(x)
                            return x

                    model = SimpleCNN(num_classes=len(SKIN_CLASSES))
                    model.load_state_dict(state_dict)
                    MODEL = model
                except Exception as e_simple:
                    raise RuntimeError("Unable to load model from checkpoint. Tried ResNet and a fallback CNN.") from e_simple
    else:
        raise RuntimeError("Unrecognized checkpoint format for model file.")

    MODEL.to(device)
    MODEL.eval()
    print(f"✅ Model loaded successfully on {device}")

# ------------------------
# 4. Signup/Login Endpoints
# ------------------------
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json() or {}
    username = data.get('name') or data.get('username')
    email = data.get('email')
    password = data.get('password')
    phone = data.get('phone')  # <-- ADDED: Get phone from request

    if not all([username, email, password]):
        return jsonify({"status": "error", "message": "Missing fields"}), 400

    # check duplicates
    if User.query.filter((User.username == username) | (User.email == email)).first():
        return jsonify({"status": "error", "message": "Username or email already exists"}), 409

    hashed = bcrypt.generate_password_hash(password).decode('utf-8')
    # <-- ADDED: 'phone=phone' to save it in the database
    user = User(username=username, email=email, phone=phone, password_hash=hashed)
    
    try:
        db.session.add(user)
        db.session.commit()
        return jsonify({"status": "success", "message": "User registered", "user_id": user.id}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    email = data.get('email')
    password = data.get('password')
    if not all([email, password]):
        return jsonify({"status":"error","message":"Missing fields"}), 400

    user = User.query.filter_by(email=email).first()
    if user and bcrypt.check_password_hash(user.password_hash, password):
        return jsonify({"status":"success","user_id":user.id, "username": user.username})
    else:
        return jsonify({"status":"error","message":"Invalid credentials"}), 401

# In app.py, add this entire new function

@app.route('/api/selftest')
def selftest():
    # This is a URL to a standard, high-quality face image
    test_image_url = "https://www.biometricupdate.com/wp-content/uploads/2022/08/face-biometrics-on-a-woman-at-a-un-refugee-camp-scaled.jpg"
    
    try:
        print("--- RUNNING SELF-TEST ---")
        # Download the image
        response = requests.get(test_image_url)
        filestr = response.content
        npimg = np.frombuffer(filestr, np.uint8)
        image_cv = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if image_cv is None:
            return jsonify({"status": "error", "message": "SELF-TEST FAILED: Could not decode test image."}), 500

        # Run the exact same face detection logic
        (h, w) = image_cv.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image_cv, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        highest_confidence = np.max(detections[0, 0, :, 2])

        print(f"--- SELF-TEST RESULT --- Confidence: {highest_confidence:.4f}")

        if highest_confidence > 0.3:
            return jsonify({
                "status": "success",
                "message": "SELF-TEST PASSED: The face detector is working correctly on the server.",
                "confidence": f"{highest_confidence:.4f}"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "SELF-TEST FAILED: The face detector is NOT working on the server.",
                "confidence": f"{highest_confidence:.4f}"
            }), 500

    except Exception as e:
        print(f"--- SELF-TEST CRASHED --- Error: {e}")
        return jsonify({"status": "error", "message": f"An exception occurred during self-test: {e}"}), 500

# ------------------------
# 5. Prediction Function (accepts PIL.Image or path)
# ------------------------
def predict_skin_type(image_input):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if isinstance(image_input, Image.Image):
        image = image_input
    else:
        image = Image.open(image_input).convert('RGB')

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = MODEL(img_tensor)
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    predicted_idx = int(probs.argmax())
    skin_type = SKIN_CLASSES[predicted_idx]
    confidence = float(probs[predicted_idx] * 100)

    detailed_scores = [{"type": cls, "score": float(prob)} for cls, prob in zip(SKIN_CLASSES, probs)]

    return {
        "skin_type": skin_type,
        "confidence": f"{confidence:.1f}",
        "detailed_scores": detailed_scores
    }

# ------------------------
# 6. API Endpoint: /predict (improved - reads file.stream)
# ------------------------

@app.route('/api/predict', methods=['POST'])
def predict():
    user_id = request.form.get('user_id')
    file = request.files.get('image')

    if not user_id or not file:
        return jsonify({"status": "error", "message": "Missing user_id or image"}), 400

    filestr = file.read()
    
    # --- THIS IS THE LINE THAT IS FIXED ---
    # We replace the old np.fromstring with the modern np.frombuffer
    npimg = np.frombuffer(filestr, np.uint8)
   
    
    image_cv = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Check if image was decoded successfully
    if image_cv is None:
        return jsonify({"status": "error", "message": "Could not decode image. File may be corrupt."}), 400

    (h, w) = image_cv.shape[:2]

    # DNN Detection Logic
    blob = cv2.dnn.blobFromImage(cv2.resize(image_cv, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    best_detection_index = np.argmax(detections[0, 0, :, 2])
    highest_confidence = detections[0, 0, best_detection_index, 2]

    print(f"!!! FINAL CHECK !!! Highest confidence detected: {highest_confidence:.4f}")

    if highest_confidence < 0.2:
        return jsonify({"status": "error", "message": "No face detected. Please upload a clear photo of your face."}), 400

    box = detections[0, 0, best_detection_index, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    face_roi = image_cv[startY:endY, startX:endX]
    face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))

    try:
        result = predict_skin_type(face_pil)
        
        try:
            uid = int(user_id)
            user = User.query.get(uid)
            if user:
                scan = Scan(user_id=uid, skin_type=result['skin_type'], confidence=float(result['confidence']))
                db.session.add(scan)
                db.session.commit()
        except Exception as e:
            db.session.rollback()
            print("Warning: failed to save automatic scan:", e)

        return jsonify({"status": "success", **result}), 200
    except Exception as e:
        print("Prediction failed:", e)
        return jsonify({"status": "error", "message": f"Prediction failed: {e}"}), 500
# ------------------------
# 7. API Endpoint: /scan/save (explicit)
# ------------------------
@app.route('/api/scan/save', methods=['POST'])
def save_scan():
    data = request.get_json() or {}
    user_id = data.get('user_id')
    skin_type = data.get('skin_type')
    confidence = data.get('confidence')

    if not all([user_id is not None, skin_type, confidence is not None]):
        return jsonify({"status": "error", "message": "Missing data"}), 400

    try:
        scan = Scan(user_id=int(user_id), skin_type=skin_type, confidence=float(confidence))
        db.session.add(scan)
        db.session.commit()
        return jsonify({"status": "success", "message": "Scan saved successfully"}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": f"Failed to save scan: {e}"}), 500

# ------------------------
# 8. Profile endpoints & avatar serving
# ------------------------
@app.route('/api/user/<int:user_id>', methods=['GET'])
def get_user_profile(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({"status":"error","message":"User not found"}), 404

    avatar_url = None
    if user.avatar:
        avatar_url = f"/uploads/{user.avatar}"

    return jsonify({
        "status":"success",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "avatar": avatar_url,
            "phone": user.phone,
            "created_at": user.created_at.strftime("%Y-%m-%d")
        }
    }), 200

@app.route('/api/user/<int:user_id>', methods=['PUT'])
def update_user_profile(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({"status":"error","message":"User not found"}), 404

    data = request.get_json() or {}
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    # basic uniqueness checks
    if username and username != user.username:
        if User.query.filter_by(username=username).first():
            return jsonify({"status":"error","message":"Username already taken"}), 409
        user.username = username
    if email and email != user.email:
        if User.query.filter_by(email=email).first():
            return jsonify({"status":"error","message":"Email already taken"}), 409
        user.email = email
    if password:
        user.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    try:
        db.session.commit()
        return jsonify({"status":"success","message":"Profile updated"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"status":"error","message":str(e)}), 500

@app.route('/api/user/<int:user_id>/avatar', methods=['POST'])
def upload_avatar(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({"status":"error","message":"User not found"}), 404

    file = request.files.get('avatar')
    if not file:
        return jsonify({"status":"error","message":"No file provided"}), 400

    filename = secure_filename(file.filename)
    if filename == '':
        return jsonify({"status":"error","message":"Invalid filename"}), 400

    timestamp = int(time.time())
    filename = f"user_{user_id}_{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(filepath)
        user.avatar = filename
        db.session.commit()
        return jsonify({"status":"success","avatar_url": f"/uploads/{filename}"}), 200
    except Exception as e:
        db.session.rollback()
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"status":"error","message": f"Failed to save avatar: {e}"}), 500

@app.route('/uploads/<path:filename>', methods=['GET'])
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ------------------------
# 9. User history endpoint
# ------------------------
@app.route('/api/user/history/<int:user_id>', methods=['GET'])
def get_user_history(user_id):
    scans = Scan.query.filter_by(user_id=user_id).order_by(Scan.timestamp.desc()).all()
    history = [
        {
            "id": s.id,
            "skin_type": s.skin_type,
            "confidence": s.confidence,
            "timestamp": s.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        } for s in scans
    ]
    return jsonify({"status":"success", "history": history}), 200

@app.route('/')
def index():
    return send_from_directory('static', 'intro.html') 
@app.route('/<path:path>')
def serve_static_files(path):
    # This serves all the other files like login.html, style.css, etc.
    return send_from_directory('static', path)

# ------------------------
# 10. Run App
# ------------------------
if __name__ == '__main__':
    load_model()  
    app.run(host='0.0.0.0', port=5000, debug=True)
