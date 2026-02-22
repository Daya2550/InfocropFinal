from flask import Flask, render_template, redirect, url_for, flash, request, send_file, send_from_directory, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from config import Config
from models import db, User, ServiceRequest, ChatMessage, Feedback
from werkzeug.security import generate_password_hash
from werkzeug.utils import secure_filename
import uuid
import os
from xhtml2pdf import pisa
from io import BytesIO
import io
from datetime import datetime
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
# Removed redundant/conflicting genai import
import pandas as pd
import cv2
import joblib
import traceback
from collections import Counter
from crop_database import get_crop_info, analyze_input_fit
from pdf_generator import CropReportGenerator
import json
from farm_planner import farm_bp
from flask.json.provider import DefaultJSONProvider
import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
import pandas as pd
import joblib
# Load environment variables of whwerPrediction
load_dotenv()


class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if hasattr(obj, 'item'): # For numpy scalars like np.str_
            return obj.item()
        return super().default(obj)

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'doc', 'docx', 'txt'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

app = Flask(__name__)
app.config.from_object(Config)
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)


db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ── Farm Planner Blueprint ─────────────────────────────────
app.register_blueprint(farm_bp)

# ── Jinja2 filter: render Gemini markdown text as HTML ─────
import re as _re

@app.template_filter('render_ai_html')
def render_ai_html(text):
    """Convert Gemini markdown response to clean HTML — no raw ** or ## symbols."""
    if not text:
        return ''
    lines = text.split('\n')
    html_parts = []
    for raw in lines:
        line = raw.strip()
        if not line:
            html_parts.append('<div style="height:6px"></div>')
            continue
        # Horizontal rules
        if _re.match(r'^[-=\u2550\u2500]{3,}$', line):
            html_parts.append('<hr style="border:none;border-top:1px solid #c8e6c9;margin:8px 0;">')
            continue
        # ## or # Headings
        if _re.match(r'^#{1,3}\s+', line):
            content = _re.sub(r'^#{1,3}\s+', '', line)
            content = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
            html_parts.append(f'<div class="ai-h">{content}</div>')
            continue
        # **1. SECTION** or **ALL_CAPS:** bold headers
        if _re.match(r'^\*\*[\d]+[\.\)]\s*.+\*\*\s*$', line) \
                or _re.match(r'^\*\*[A-Z][A-Z_\s]+[:\s]*\*\*\s*$', line):
            content = line.replace('**', '').strip()
            html_parts.append(f'<div class="ai-h">{content}</div>')
            continue
        # **Label:** text  (inline bold label)
        if line.startswith('**') and ':**' in line:
            content = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
            content = _re.sub(r'\*([^*]+?)\*', r'<em>\1</em>', content)
            html_parts.append(f'<div class="ai-h">{content}</div>')
            continue
        # Bullet list: - / * / bullet
        if _re.match(r'^[-\u2022]\s+', line) or (line.startswith('* ') and not line.startswith('**')):
            content = _re.sub(r'^[-*\u2022]\s+', '', line)
            content = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
            content = _re.sub(r'\*([^*]+?)\*', r'<em>\1</em>', content)
            html_parts.append(f'<div class="ai-li">\u25b8\u00a0{content}</div>')
            continue
        # Numbered list: 1. / 2)
        m = _re.match(r'^(\d+[\.\)])\s+(.*)', line)
        if m:
            num, content = m.group(1), m.group(2)
            content = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
            content = _re.sub(r'\*([^*]+?)\*', r'<em>\1</em>', content)
            html_parts.append(f'<div class="ai-li"><span class="ai-num">{num}</span>\u00a0{content}</div>')
            continue
        # Plain paragraph
        line = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
        line = _re.sub(r'\*([^*]+?)\*', r'<em>\1</em>', line)
        html_parts.append(f'<p class="ai-p">{line}</p>')
    return '\n'.join(html_parts)


# Ensure upload directory exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# AI Model Initialization
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')
cnn_model = CNN.CNN(39)
cnn_model_path = "plant_disease_model_1_latest.pt"

if os.path.exists(cnn_model_path):
    try:
        cnn_model.load_state_dict(torch.load(cnn_model_path))
        cnn_model.eval()
        print(f"Successfully loaded CNN model from {cnn_model_path}")
    except Exception as e:
        print(f"Error loading CNN model: {e}")
else:
    print(f"\n{'!'*60}\nCRITICAL ERROR: Model file '{cnn_model_path}' NOT FOUND!\n{'!'*60}\n")

pdf_gen = CropReportGenerator()

# Ensemble model paths
ENSEMBLE_MODEL_PATHS = {
    'random_forest': 'random_forest_model.pkl',
    'gradient_boosting': 'gradient_boosting_model.pkl',
    'decision_tree': 'decision_tree_model.pkl'
}
ensemble_models = {}

# Input validation ranges
VALIDATION_RANGES = {
    'nitrogen': (0, 150), 'phosphorus': (0, 150), 'potassium': (0, 150),
    'temperature': (0, 60), 'humidity': (0, 100), 'ph': (0, 14), 'rainfall': (0, 500)
}

STRICT_RULES = {
    'high_temp_low_humidity': {
        'condition': lambda d: d['temperature'] > 35 and d['humidity'] < 30,
        'warning': 'Extremely high temperature with very low humidity is unusual'
    },
    'low_temp_high_rainfall': {
        'condition': lambda d: d['temperature'] < 10 and d['rainfall'] > 300,
        'warning': 'Very low temperature with high rainfall is uncommon'
    },
    'extreme_ph': {
        'condition': lambda d: d['ph'] < 4 or d['ph'] > 9,
        'warning': 'Extreme pH values may affect crop viability'
    },
    'nutrient_imbalance': {
        'condition': lambda d: max(d['nitrogen'], d['phosphorus'], d['potassium']) > 3 * min(d['nitrogen'], d['phosphorus'], d['potassium']),
        'warning': 'Significant nutrient imbalance detected'
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- AI Helper Functions ---
def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = cnn_model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

def load_ensemble_models():
    """Load all trained ML models for ensemble prediction."""
    global ensemble_models
    success_count = 0
    print("\n" + "=" * 80 + "\nLOADING ENSEMBLE MODELS\n" + "=" * 80)
    for model_name, model_path in ENSEMBLE_MODEL_PATHS.items():
        try:
            if os.path.exists(model_path):
                ensemble_models[model_name] = joblib.load(model_path)
                print(f"[OK] Loaded {model_name.replace('_', ' ').title()}")
                success_count += 1
            else:
                print(f"[ERROR] Model file not found: {model_path}")
        except Exception as e:
            print(f"[ERROR] Error loading {model_name}: {e}")
    print(f"\n[OK] Successfully loaded {success_count}/{len(ENSEMBLE_MODEL_PATHS)} models\n" + "=" * 80)
    return success_count == len(ENSEMBLE_MODEL_PATHS)

def validate_input(data):
    param_names = {
        'nitrogen': 'Nitrogen', 'phosphorus': 'Phosphorus', 'potassium': 'Potassium',
        'temperature': 'Temperature', 'humidity': 'Humidity', 'ph': 'pH', 'rainfall': 'Rainfall'
    }
    validated = {}
    warnings_list = []
    for param, (min_val, max_val) in VALIDATION_RANGES.items():
        try:
            value = float(data.get(param, ''))
            if not min_val <= value <= max_val:
                return False, f"Invalid {param_names[param]}: must be between {min_val} and {max_val}", None, []
            validated[param] = value
        except (ValueError, TypeError):
            return False, f"Invalid {param_names[param]}: must be a valid number", None, []
    for rule_name, rule in STRICT_RULES.items():
        if rule['condition'](validated):
            warnings_list.append(rule['warning'])
    return True, "Valid", validated, warnings_list

def predict_ensemble(input_data, user_inputs_dict):
    predictions = {}
    model_probs = {}
    for model_name, model in ensemble_models.items():
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)
        confidence = float(np.max(proba)) * 100
        predictions[model_name] = {'crop': pred, 'confidence': round(confidence, 2)}
        model_probs[model_name] = proba[0]
        classes = model.classes_
    
    crops = [p['crop'] for p in predictions.values()]
    vote_counts = Counter(crops)
    most_common_items = vote_counts.most_common()
    most_common_crop, vote_count = most_common_items[0]
    is_tie_breaker = False
    
    if vote_count == 1 and len(crops) > 1:
        is_tie_breaker = True
        soft_vote_scores = {}
        for i, crop_label in enumerate(classes):
            total_prob = sum(model_probs[m][i] for m in ensemble_models.keys())
            soft_vote_scores[crop_label] = total_prob
        most_common_crop = max(soft_vote_scores, key=soft_vote_scores.get)
        vote_count = "Tie-Breaker"
    
    agreement = float((vote_count / len(crops)) * 100) if isinstance(vote_count, int) else 0.0
    avg_confidence = float(np.mean([p['confidence'] for p in predictions.values()]))
    feature_names = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    feature_importance = {}
    
    if 'random_forest' in ensemble_models:
        rf_model = ensemble_models['random_forest']
        importances = rf_model.feature_importances_
        feature_importance = {feature_names[i]: round(float(importances[i]) * 100, 2) for i in range(len(feature_names))}
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:3]
    else:
        top_features = []
        sorted_features = []
    
    crop_info = get_crop_info(most_common_crop)
    input_analysis = analyze_input_fit(most_common_crop, user_inputs_dict)
    reasoning = generate_reasoning(most_common_crop, top_features, input_analysis, user_inputs_dict)
    
    return {
        'final_crop': str(most_common_crop),
        'confidence': float(avg_confidence),
        'agreement': float(agreement) if isinstance(agreement, (int, float, np.floating)) else agreement,
        'vote_count': str(f"{vote_count}/{len(crops)}" if isinstance(vote_count, int) else vote_count),
        'individual_predictions': {str(k): {'crop': str(v['crop']), 'confidence': float(v['confidence'])} for k, v in predictions.items()},
        'unanimous': bool(vote_count == len(crops) if isinstance(vote_count, int) else False),
        'is_tie_breaker': bool(is_tie_breaker),
        'feature_importance': {str(k): float(v) for k, v in dict(sorted_features[:5]).items()},
        'top_influencing_factors': [{'factor': str(f[0]), 'importance': float(f[1])} for f in top_features],
        'crop_info': crop_info,
        'input_analysis': input_analysis,
        'reasoning': [str(r) for r in reasoning]
    }

def generate_reasoning(crop_name, top_features, input_analysis, user_inputs):
    reasoning = [f"{crop_name.title()} is recommended based on your soil and weather conditions."]
    if top_features:
        top_factor = top_features[0]
        reasoning.append(f"The most influential factor is {top_factor[0]} ({top_factor[1]}% importance), with your value of {user_inputs.get(top_factor[0].lower(), 'N/A')}.")
    if input_analysis['match_percentage'] != 'N/A':
        match_pct = input_analysis['match_percentage']
        if match_pct >= 80: reasoning.append(f"Your conditions match {match_pct}% with optimal {crop_name} requirements - Excellent fit!")
        elif match_pct >= 60: reasoning.append(f"Your conditions match {match_pct}% with optimal {crop_name} requirements - Good fit.")
        else: reasoning.append(f"Your conditions match {match_pct}% with optimal {crop_name} requirements.")
    if input_analysis['matching_factors']:
        reasoning.append("Favorable conditions: " + ", ".join(input_analysis['matching_factors'][:2]))
    return reasoning

# --- Routes ---



@app.route('/')
def landing_page():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/dashboard')
@login_required
def index():
    if current_user.role == 'farmer':
        return redirect(url_for('farmer_dashboard'))
    elif current_user.role == 'cs':
        return redirect(url_for('cs_dashboard'))
    elif current_user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    return redirect(url_for('login'))



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('landing_page'))
        flash('Invalid username or password')
    return render_template('auth/login.html')



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # New Fields
        name = request.form.get('name')
        surname = request.form.get('surname')
        phone = request.form.get('phone')
        age = request.form.get('age')
        address = request.form.get('address')
        email = request.form.get('email')
        
        # Handle Identity Proof Upload
        identity_file = request.files.get('identity_proof')
        identity_path = None
        if identity_file:
            filename = f"{username}_id_{identity_file.filename}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'identity_proofs')
            os.makedirs(save_path, exist_ok=True)
            identity_file.save(os.path.join(save_path, filename))
            identity_path = f"identity_proofs/{filename}"

        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
            
        new_user = User(
            username=username, 
            role='farmer',
            name=name,
            surname=surname,
            phone=phone,
            age=age,
            address=address,
            email=email,
            identity_proof=identity_path
        )
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('farmer_dashboard'))
    return render_template('auth/register.html')



@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- Integrated App2 Routes ---
@app.route('/contact')  
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('leaf.html')

# Routes for each crop/fruit (from app2.py)
@app.route('/barley')
def barley(): return render_template('barley.html')
@app.route('/maize')
def maize(): return render_template('maize.html')
@app.route('/rice')
def rice(): return render_template('rice.html')
@app.route('/soybean')
def soybean(): return render_template('soybean.html')
@app.route('/cotton')
def cotton(): return render_template('cotton.html')
@app.route('/sugarcane')
def sugarcane(): return render_template('sugarcane.html')
@app.route('/grapes')
def grapes(): return render_template('grapes.html')
@app.route('/pomegranate')
def pomegranate(): return render_template('pomegranate.html')
@app.route('/banana')
def banana(): return render_template('banana.html')
@app.route('/mango')
def mango(): return render_template('mango.html')
@app.route('/guava')
def guava(): return render_template('guava.html')
@app.route('/watermelon')
def watermelon(): return render_template('watermelon.html')
@app.route('/rose')
def rose(): return render_template('rose.html')
@app.route('/marigold')
def marigold(): return render_template('marigold.html')
@app.route('/jasmine')
def jasmine(): return render_template('jasmine.html')
@app.route('/brinjal')
def brinjal(): return render_template('brinjal.html')
@app.route('/potato')
def potato(): return render_template('potato.html')
@app.route('/chilli')
def chilli(): return render_template('chilli.html')
@app.route('/tomato')
def tomato(): return render_template('tomato.html')
@app.route('/spinach')
def spinach(): return render_template('spinach.html')
@app.route('/carrot')
def carrot(): return render_template('carrot.html')
@app.route('/aloevera')
def aloevera(): return render_template('aloevera.html')
@app.route('/amla')
def amla(): return render_template('amla.html')
@app.route('/ashwagandha')
def ashwagandha(): return render_template('ashwagandha.html')

@app.route('/pages-faq')
def faq():
    return render_template('faq.html')

@app.route('/mobile-device')     
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = secure_filename(image.filename)
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html', title=title, desc=description, prevent=prevent, 
                               image_url=image_url, pred=pred, sname=supplement_name, simage=supplement_image_url, buy_link=supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                        supplement_name=list(supplement_info['supplement name']), disease=list(disease_info['disease_name']), buy=list(supplement_info['buy link']))

@app.route('/CropRec')
def crop_rec():
    return render_template('CropRec.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        is_valid, message, validated_data, warnings = validate_input(data)
        if not is_valid:
            return jsonify({'success': False, 'error': message}), 400
        if not ensemble_models:
            return jsonify({'success': False, 'error': 'Models not loaded.'}), 500
        input_array = np.array([[validated_data['nitrogen'], validated_data['phosphorus'], validated_data['potassium'],
                                 validated_data['temperature'], validated_data['humidity'], validated_data['ph'], validated_data['rainfall']]])
        result = predict_ensemble(input_array, validated_data)
        return jsonify({'success': True, 'crop': result['final_crop'], 'confidence': result['confidence'], 'agreement': result['agreement'],
                        'vote_count': result['vote_count'], 'unanimous': result['unanimous'], 'is_tie_breaker': result['is_tie_breaker'],
                        'individual_predictions': result['individual_predictions'], 'warnings': warnings, 'feature_importance': result['feature_importance'],
                        'top_influencing_factors': result['top_influencing_factors'], 'crop_info': result['crop_info'], 'input_analysis': result['input_analysis'], 'reasoning': result['reasoning']})
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/info')
def api_info():
    return jsonify({'models_loaded': len(ensemble_models), 'model_names': list(ensemble_models.keys()), 'validation_ranges': VALIDATION_RANGES, 'strict_rules': len(STRICT_RULES)})

@app.route('/api/download_report', methods=['POST'])
def download_report():
    try:
        data = request.get_json()
        prediction_result = data.get('prediction_data')
        user_inputs = data.get('user_inputs')
        if not prediction_result or not user_inputs:
            return jsonify({'success': False, 'error': 'Missing required data'}), 400
        pdf_buffer = pdf_gen.generate_report(prediction_result, user_inputs)
        crop_name = prediction_result.get('crop', 'recommendation').lower()
        filename = f"crop_report_{crop_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(pdf_buffer, as_attachment=True, download_name=filename, mimetype='application/pdf')
    except Exception as e:
        print(f"[ERROR] PDF generation error: {e}")
        return jsonify({'success': False, 'error': 'Failed to generate PDF report'}), 500

# --- Core Logic Routes ---



@app.route('/api/create_request', methods=['POST'])
@login_required
def create_request():
    if current_user.role != 'farmer':
        return {'error': 'Unauthorized'}, 403
    
    # Check if active/pending/draft request exists
    active_req = ServiceRequest.query.filter_by(farmer_id=current_user.id, status='active').first() or \
                 ServiceRequest.query.filter_by(farmer_id=current_user.id, status='pending').first() or \
                 ServiceRequest.query.filter_by(farmer_id=current_user.id, status='draft').first()
    
    if active_req:
        return {'message': 'You main have an active request', 'request_id': active_req.id, 'token': active_req.token}

    token = str(uuid.uuid4())
    # Initial status is 'draft' so it doesn't appear in queue until message sent
    new_req = ServiceRequest(token=token, farmer_id=current_user.id, status='draft')
    db.session.add(new_req)
    db.session.commit()
    return {'message': 'Request created', 'request_id': new_req.id, 'token': token}



@app.route('/api/request/<int:request_id>/messages', methods=['GET'])
@login_required
def get_messages(request_id):
    req = ServiceRequest.query.get_or_404(request_id)
    # Access control
    if current_user.role == 'farmer' and req.farmer_id != current_user.id:
        return {'error': 'Unauthorized'}, 403
    if current_user.role == 'cs' and req.cs_id != current_user.id and req.cs_id is not None:
         return {'error': 'Unauthorized'}, 403 

    messages = ChatMessage.query.filter_by(request_id=request_id).order_by(ChatMessage.timestamp).all()
    return {
        'messages': [{
            'sender': m.sender.username,
            'role': m.sender.role,
            'message': m.message,
            'file_path': m.file_path,
            'file_name': os.path.basename(m.file_path) if m.file_path else None,
            'timestamp': m.timestamp.strftime('%H:%M'),
            'is_me': m.sender_id == current_user.id
        } for m in messages],
        'status': req.status
    }



@app.route('/api/request/<int:request_id>/send', methods=['POST'])
@login_required
def send_message(request_id):
    req = ServiceRequest.query.get_or_404(request_id)
    
    # Get message text from either JSON or form data
    message_text = None
    if request.is_json:
        data = request.get_json()
        message_text = data.get('message', '').strip() if data else ''
    else:
        message_text = request.form.get('message', '').strip()
    
    # Handle file upload if present
    file_path = None
    if 'file' in request.files:
        file = request.files['file']
        if file and file.filename and allowed_file(file.filename):
            # Check file size
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)
            
            if file_size > MAX_FILE_SIZE:
                return {'error': 'File too large. Maximum size is 5MB.'}, 400
            
            # Generate unique filename
            original_filename = secure_filename(file.filename)
            filename = f"{uuid.uuid4().hex}_{original_filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
    
    # Require either message or file
    if not message_text and not file_path:
        return {'error': 'Empty message'}, 400

    # Auto-promote 'draft' to 'pending' on first message
    if req.status == 'draft':
        req.status = 'pending'
        db.session.commit()

    # Auto-Reopen logic: If Farmer replies to a CLOSED request, reopen it as PENDING
    if req.status == 'closed' and current_user.role == 'farmer':
        req.status = 'pending'
        req.cs_id = None # Unassign to go back to general queue
        req.closed_at = None
        db.session.commit()

    # Assignment logic (auto-assign if CS replies to pending)
    if current_user.role == 'cs' and req.status == 'pending':
        req.cs_id = current_user.id
        req.status = 'active'
        db.session.commit()

    msg = ChatMessage(
        request_id=request_id,
        sender_id=current_user.id,
        message=message_text if message_text else None,
        file_path=file_path
    )
    db.session.add(msg)
    db.session.commit()
    return {'status': 'sent'}

@app.route('/api/queue', methods=['GET'])
@login_required
def get_queue():
    if current_user.role != 'cs': return {'error': 'Unauthorized'}, 403
    
    # Unassigned requests (pending OR transferred)
    pending_reqs = ServiceRequest.query.filter(ServiceRequest.status.in_(['pending', 'transferred'])).all()
    # My active requests
    my_active = ServiceRequest.query.filter_by(cs_id=current_user.id, status='active').all()
    # My closed requests (History)
    my_closed = ServiceRequest.query.filter_by(cs_id=current_user.id, status='closed').order_by(ServiceRequest.closed_at.desc()).limit(20).all()
    
    return {
        'pending': [{'id': r.id, 'farmer': r.farmer.username, 'time': r.created_at.strftime('%H:%M'), 'status': r.status} for r in pending_reqs],
        'active': [{'id': r.id, 'farmer': r.farmer.username, 'last_msg': '...'} for r in my_active],
        'closed': [{'id': r.id, 'farmer': r.farmer.username, 'closed_at': r.closed_at.strftime('%Y-%m-%d')} for r in my_closed]
    }

@app.route('/uploads/<filename>')
@login_required
def download_file(filename):
    """Serve uploaded files with permission check"""
    # Find the message containing this file
    msg = ChatMessage.query.filter(ChatMessage.file_path.contains(filename)).first()
    if not msg:
        return {'error': 'File not found'}, 404
    
    req = msg.request
    
    # Check permissions: User must be involved in the request or be admin
    if current_user.role == 'admin':
        pass  # Admin can access all files
    elif current_user.role == 'farmer' and req.farmer_id != current_user.id:
        return {'error': 'Unauthorized'}, 403
    elif current_user.role == 'cs' and req.cs_id != current_user.id:
        return {'error': 'Unauthorized'}, 403
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/farmer/history', methods=['GET'])
@login_required
def get_farmer_history():
    if current_user.role != 'farmer': return {'error': 'Unauthorized'}, 403
    
    # Fetch all requests for this farmer, ordered by latest
    requests = ServiceRequest.query.filter_by(farmer_id=current_user.id).order_by(ServiceRequest.created_at.desc()).all()
    
    return {
        'requests': [{
            'id': r.id,
            'status': r.status,
            'created_at': r.created_at.strftime('%Y-%m-%d'),
            'closed_at': r.closed_at.strftime('%Y-%m-%d') if r.closed_at else None,
            'cs_agent': r.cs_agent.username if r.cs_agent else 'Unassigned'
        } for r in requests]
    }

@app.route('/api/request/<int:request_id>/close', methods=['POST'])
@login_required
def close_request(request_id):
    print(f"DEBUG: Closing request {request_id} by {current_user.username}")
    req = ServiceRequest.query.get_or_404(request_id)
    
    # Allow Farmer (owner) or Admin to close. CS CANNOT close.
    if current_user.role == 'cs':
        return {'error': 'CS Agents cannot permanently close requests. Use Transfer instead.'}, 403
        
    if current_user.role == 'farmer' and req.farmer_id != current_user.id:
        print(f"DEBUG: Unauthorized close attempt by farmer {current_user.id}")
        return {'error': 'Unauthorized'}, 403
        
    # Admin is allowed implicitly by not being restricted above
        
    if req.status == 'closed': 
        print(f"DEBUG: Request {request_id} already closed")
        return {'status': 'already_closed'}
    
    req.status = 'closed'
    req.closed_at = datetime.utcnow()
    db.session.commit()
    print(f"DEBUG: Request {request_id} closed successfully")
    return {'status': 'closed'}

@app.route('/api/request/<int:request_id>/accept', methods=['POST'])
@login_required
def accept_request(request_id):
    if current_user.role != 'cs': return {'error': 'Unauthorized'}, 403
    req = ServiceRequest.query.get_or_404(request_id)
    if req.status not in ['pending', 'transferred']: return {'error': 'Request not available'}, 400
    
    req.cs_id = current_user.id
    req.status = 'active'
    db.session.commit()
    print(f"DEBUG: Request {request_id} accepted by {current_user.username}")
    return {'status': 'accepted'}

@app.route('/api/request/<int:request_id>/feedback', methods=['POST'])
@login_required
def submit_feedback(request_id):
    if current_user.role != 'farmer': return {'error': 'Unauthorized'}, 403
    req = ServiceRequest.query.get_or_404(request_id)
    
    # Remove the requirement that request must be closed
    # Now accepting feedback will also close the request
    
    data = request.json
    feedback = Feedback(
        request_id=request_id,
        rating=data.get('rating'),
        comments=data.get('comments')
    )
    db.session.add(feedback)
    
    # Close the request after feedback submission
    if req.status != 'closed':
        req.status = 'closed'
        req.closed_at = datetime.utcnow()
    
    db.session.commit()
    return {'status': 'feedback_received_and_closed'}

@app.route('/api/request/<int:request_id>/transfer', methods=['POST'])
@login_required
def transfer_request(request_id):
    # Allow CS or Admin to transfer/release
    if current_user.role not in ['cs', 'admin']: 
        return {'error': 'Unauthorized'}, 403
    
    req = ServiceRequest.query.get_or_404(request_id)
    print(f"DEBUG: Transferring request {request_id} by {current_user.username}")
    
    # improved transfer logic: explicitly unassign or reassign
    req.cs_id = None
    req.status = 'transferred'
    db.session.commit()
    print(f"DEBUG: Request {request_id} transferred")
    return {'status': 'transferred'}

# --- Dashboard Routes ---

@app.route('/dashboard/farmer')
@login_required
def farmer_dashboard():
    if current_user.role != 'farmer': return redirect(url_for('index'))
    return render_template('dashboard/farmer.html')

@app.route('/dashboard/cs')
@login_required
def cs_dashboard():
    if current_user.role != 'cs': return redirect(url_for('index'))
    return render_template('dashboard/cs.html')

@app.route('/dashboard/admin')
@login_required
def admin_dashboard():
    if current_user.role != 'admin': return redirect(url_for('index'))
    
    # Stats
    active_count = ServiceRequest.query.filter_by(status='active').count()
    closed_count = ServiceRequest.query.filter_by(status='closed').count()
    cs_online = User.query.filter_by(role='cs').count() 
    
    # Categorized Requests
    solved_requests = ServiceRequest.query.filter_by(status='closed').order_by(ServiceRequest.closed_at.desc()).all()
    unsolved_requests = ServiceRequest.query.filter_by(status='active').order_by(ServiceRequest.created_at.desc()).all()
    transferred_requests = ServiceRequest.query.filter_by(status='transferred').order_by(ServiceRequest.created_at.desc()).all()
    pending_requests = ServiceRequest.query.filter_by(status='pending').order_by(ServiceRequest.created_at.desc()).all()
    
    # Staff (CS Agents + Admins)
    staff_users = User.query.filter(User.role.in_(['cs', 'admin'])).all()
    cs_agents = [u for u in staff_users if u.role == 'cs']
    admins = [u for u in staff_users if u.role == 'admin']
    
    # Feedback Analytics
    all_feedbacks = db.session.query(Feedback, ServiceRequest, User).join(
        ServiceRequest, Feedback.request_id == ServiceRequest.id
    ).join(
        User, ServiceRequest.farmer_id == User.id
    ).order_by(Feedback.created_at.desc()).all()
    
    # CS Agent Performance Metrics
    cs_performance = []
    for agent in cs_agents:
        # Get all closed requests handled by this agent that have feedback
        agent_requests = ServiceRequest.query.filter_by(cs_id=agent.id, status='closed').all()
        feedbacks_for_agent = [req.feedback for req in agent_requests if req.feedback]
        
        if feedbacks_for_agent:
            avg_rating = sum(f.rating for f in feedbacks_for_agent) / len(feedbacks_for_agent)
            total_feedback = len(feedbacks_for_agent)
        else:
            avg_rating = 0
            total_feedback = 0
        
        cs_performance.append({
            'agent': agent,
            'avg_rating': round(avg_rating, 2),
            'total_feedback': total_feedback,
            'total_resolved': len(agent_requests)
        })
    
    # Sort by average rating descending
    cs_performance.sort(key=lambda x: x['avg_rating'], reverse=True)

    return render_template('dashboard/admin.html', 
                           active_requests=active_count,
                           total_resolved=closed_count,
                           cs_online=cs_online,
                           solved_requests=solved_requests,
                           unsolved_requests=unsolved_requests,
                           transferred_requests=transferred_requests,
                           pending_requests=pending_requests,
                           cs_agents=cs_agents,
                           staff_users=staff_users,
                           all_feedbacks=all_feedbacks,
                           cs_performance=cs_performance)

# --- Admin Action Routes ---

@app.route('/admin/create_staff', methods=['POST'])
@login_required
def create_staff_user():
    if current_user.role != 'admin': return {'error': 'Unauthorized'}, 403
    
    username = request.form.get('username')
    password = request.form.get('password')
    email = request.form.get('email')
    role = request.form.get('role', 'cs')
    
    # Only root admin 'admin' can create other admins
    if role == 'admin' and current_user.username != 'admin':
        flash('Only the root admin can create other admins', 'error')
        return redirect(url_for('admin_dashboard'))

    if User.query.filter_by(username=username).first():
        flash('Username already exists', 'error')
        return redirect(url_for('admin_dashboard'))

    new_staff = User(username=username, email=email, role=role)
    new_staff.set_password(password)
    db.session.add(new_staff)
    db.session.commit()
    
    flash(f'Staff User ({role}) created successfully', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/staff_user/<int:user_id>/edit', methods=['POST'])
@login_required
def edit_staff_user(user_id):
    if current_user.role != 'admin': return {'error': 'Unauthorized'}, 403
    
    user = User.query.get_or_404(user_id)
    if user.username == 'admin':
        flash('The root admin profile cannot be edited', 'error')
        return redirect(url_for('admin_dashboard'))
        
    if user.role not in ['cs', 'admin']: return {'error': 'Cannot edit non-staff user'}, 400
    
    # Root admin check for role changes
    new_role = request.form.get('role')
    if new_role and new_role != user.role:
        if current_user.username != 'admin':
            flash('Only the root admin can change user roles', 'error')
            return redirect(url_for('admin_dashboard'))
        user.role = new_role
        
    user.username = request.form.get('username')
    user.email = request.form.get('email')
    
    password = request.form.get('password')
    if password:
        user.set_password(password)
        
    db.session.commit()
    flash('Staff User updated', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/request/<int:request_id>/edit', methods=['POST'])
@login_required
def edit_request(request_id):
    if current_user.role != 'admin': return {'error': 'Unauthorized'}, 403
    
    req = ServiceRequest.query.get_or_404(request_id)
    
    # Update Status
    new_status = request.form.get('status')
    if new_status in ['pending', 'active', 'closed', 'transferred']:
        req.status = new_status
        if new_status == 'closed':
            req.closed_at = datetime.utcnow()
        elif new_status == 'pending':
            req.cs_id = None # Unassign if pending
            
    # Update Assignee (Optional)
    cs_id = request.form.get('cs_id')
    if cs_id:
        req.cs_id = int(cs_id)
    elif cs_id == '':
        req.cs_id = None

    db.session.commit()
    flash('Request updated', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/report/<int:request_id>/pdf')
@login_required
def generate_pdf(request_id):
    if current_user.role != 'admin': return {'error': 'Unauthorized'}, 403
    
    req = ServiceRequest.query.get_or_404(request_id)
    html = render_template('pdf_report.html', request=req, generated_at=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
    
    pdf = BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=pdf)
    
    if pisa_status.err:
        return 'We had some errors <pre>' + html + '</pre>'
        
    pdf.seek(0)
    
    return send_file(pdf, as_attachment=True, download_name=f'report_{req.id}.pdf', mimetype='application/pdf')

# Temporary init helper
@app.route('/init_db')
def init_db():
    db.create_all()
    # Create admin if not exists
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', role='admin')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
    return "Database initialized!"














#Whather Preddiction

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the model
# Initialize the models with unique names to avoid conflicts with ML model
weather_model = genai.GenerativeModel('gemini-3-flash-preview')
chat_model = genai.GenerativeModel('gemini-3-flash-preview')

@app.route('/WhetherPrediction')
def weather_prediction():
    return render_template('WhetherPrediction.html')

@app.route('/api/get_analysis', methods=['POST'])
def get_analysis():
    try:
        data = request.json
        city = data.get('city')
        state = data.get('state')
        crop = data.get('crop')
        season = data.get('season')
        extra_info = data.get('extra_info', '')

        if not all([city, state, crop, season]):
            return jsonify({'error': 'Missing required fields (City, State, Crop, Season)'}), 400

        # Construct the prompt
        prompt = f"""
        Act as an agricultural expert. 
        Detailed Context:
        - Location: {city}, {state}
        - Crop: {crop}
        - Season: {season}
        - Additional Info: {extra_info}

        Task: Provide a concise weather forecast analysis and crop advice based on general climate knowledge for this region and season. 
        Focus ONLY on the main points (Weather Outlook, Crop Suitability, Key Actions ,profitable or Not with that crop ,Other Alternative Crops). 
        Do not provide any preamble or extra fluff. Keep it under 150 words if possible.main go get the whether info from real time and the 
        """

        # Generate response using renamed weather_model
        response = weather_model.generate_content(prompt)
        
        return jsonify({'result': response.text})

    except ResourceExhausted as e:
        print(f"Quota Exceeded: {e}")
        return jsonify({'error': 'AI Quota Exceeded. Please wait 1 minute and try again.'}), 429
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500









#/PricePrediction model





# Load model and artifacts
def load_model():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Base Directory: {base_dir}")
        model_path = os.path.join(base_dir, "model.pkl")
        encoder_path = os.path.join(base_dir, "encoder.pkl")
        columns_path = os.path.join(base_dir, "columns.pkl")
        
        print(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        print(f"Loading encoder from: {encoder_path}")
        encoders = joblib.load(encoder_path)
        print(f"Loading columns from: {columns_path}")
        model_columns = joblib.load(columns_path)
        return model, encoders, model_columns
    except FileNotFoundError as e:
        print(f"Error loading model (File Not Found): {e}")
        return None, None, None
    except Exception as e:
        print(f"Unexpected error loading model: {e}")
        # traceback.print_exc() # detailed trace
        return None, None, None

model, encoders, model_columns = load_model()

# Load dataset for statistics (cached)
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "crop_price_dataset.csv")
    print(f"Loading dataset from: {csv_path}")
    dataset = pd.read_csv(csv_path)
except FileNotFoundError:
    print("Dataset not found at:", os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_price_dataset.csv"))
    dataset = pd.DataFrame()
except Exception as e:
    print(f"Error loading dataset: {e}")
    dataset = pd.DataFrame()

@app.route("/PricePrediction")
def PricePrediction():
    return render_template("PricePrediction.html")

@app.route("/get_averages", methods=["POST"])
def get_averages():
    try:
        data = request.get_json()
        crop = data.get("crop")
        state = data.get("state")
        market = data.get("market")
        season = data.get("season")
        
        # Clean inputs
        if crop: crop = crop.strip()
        if state: state = state.strip()
        if market: market = market.strip()
        if season: season = season.strip()
        
        if dataset.empty:
             return jsonify({"error": "Dataset not available"})

        # Filter dataset
        filtered_df = dataset.copy()
        if crop and crop != 'Select':
            filtered_df = filtered_df[filtered_df['Crop_Name'] == crop]
        if state and state != 'Select':
            filtered_df = filtered_df[filtered_df['State'] == state]
        if market:
             filtered_df = filtered_df[filtered_df['Market_Name'] == market]
        if season:
             filtered_df = filtered_df[filtered_df['Season'] == season]
             
        if filtered_df.empty:
            return jsonify({"error": "No data found for this combination"})
            
        # Calculate averages for specific fields
        averages = {
            "prev_price": round(filtered_df["Previous_Day_Price"].mean(), 2),
            "arrival": round(filtered_df["Market_Arrival_Quantity"].mean(), 2),
            "rainfall": round(filtered_df["Rainfall"].mean(), 2),
            "temp_max": round(filtered_df["Temperature_Max"].mean(), 2),
            "temp_min": round(filtered_df["Temperature_Min"].mean(), 2),
            "humidity": round(filtered_df["Humidity"].mean(), 2)
        }
        
        return jsonify(averages)
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/get_locations", methods=["GET"])
def get_locations():
    try:
        if dataset.empty:
             return jsonify({})
        
        # Build hierarchy: State -> District -> Market
        locations = {}
        
        states = dataset['State'].unique()
        for state in states:
            state_df = dataset[dataset['State'] == state]
            districts = state_df['District'].unique()
            
            locations[state] = {}
            for district in districts:
                dist_df = state_df[state_df['District'] == district]
                markets = dist_df['Market_Name'].unique().tolist()
                locations[state][district] = markets
                
        return jsonify(locations)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/predict", methods=["POST"])
def predict():
    global model, encoders, model_columns
    if not model:
        model, encoders, model_columns = load_model()
    if not model:
        return render_template("PricePrediction.html", error="Model not loaded. Please run train_price_model.py first.")

    try:
        print("=" * 60)
        print("NEW PREDICTION REQUEST")
        print("=" * 60)

        # ── Helpers ──────────────────────────────────────────────
        def get_float(key, default=None):
            val = request.form.get(key, '').strip()
            if not val: return default
            try: return float(val)
            except ValueError: return default

        def get_int(key, default=None):
            val = request.form.get(key, '').strip()
            if not val: return default
            try: return int(float(val))
            except ValueError: return default

        # ── Validate required fields ──────────────────────────────
        required_fields = ["date", "crop", "state", "district", "market",
                           "season", "prev_price", "prev_7_price", "prev_30_price"]
        for field in required_fields:
            if not request.form.get(field, '').strip():
                raise ValueError(f"Missing required field: '{field}'")

        # ── Parse basic fields ────────────────────────────────────
        crop_name = request.form.get("crop", "").strip()
        state     = request.form.get("state", "").strip()
        district  = request.form.get("district", "").strip()
        market    = request.form.get("market", "").strip()
        season    = request.form.get("season", "").strip()
        date_str  = request.form.get("date", "").strip()

        festival_flag = 1 if request.form.get("festival", "No") in ['Yes', '1'] else 0
        harvest_flag  = 1 if request.form.get("harvest_season", "No") in ['Yes', '1'] else 0

        # ── Price fields ──────────────────────────────────────────
        p_day = get_float("prev_price", 0) or 0
        p_7   = get_float("prev_7_price", 0) or 0
        p_30  = get_float("prev_30_price", 0) or 0

        # Auto-calculate Price_Trend
        if p_7 > 0:
            ratio = p_day / p_7
            price_trend = "Increasing" if ratio > 1.05 else ("Decreasing" if ratio < 0.95 else "Stable")
        else:
            price_trend = "Stable"
        print(f"Auto Price_Trend: {price_trend} (Day=₹{p_day}, 7d=₹{p_7}, 30d=₹{p_30})")

        # ── Date features ─────────────────────────────────────────
        dt      = pd.to_datetime(date_str)
        day     = int(dt.day)
        month   = int(dt.month)
        year    = int(dt.year)
        week_no = int(dt.isocalendar().week)

        # ── Dataset medians for optional fields ───────────────────
        ds_medians = {} if dataset.empty else {
            col: float(pd.to_numeric(dataset[col], errors='coerce').median())
            for col in dataset.select_dtypes(include=[np.number]).columns
        }

        def fv(key, form_key, default=0.0):
            v = get_float(form_key)
            return v if v is not None else ds_medians.get(key, default)

        # ── Build ALL 31 features ─────────────────────────────────
        rainfall       = fv("Rainfall",              "rainfall")
        temp_max       = fv("Temperature_Max",        "temp_max")
        temp_min       = fv("Temperature_Min",        "temp_min")
        humidity       = fv("Humidity",               "humidity")
        arrival        = fv("Market_Arrival_Quantity","arrival")
        production     = fv("Production_Quantity",    "production")
        yield_ha       = fv("Yield_Per_Hectare",      "yield_hectare")
        stock          = fv("Stock_Available",        "stock")
        soil_moisture  = fv("Soil_Moisture",          "soil_moisture")
        demand_index   = fv("Demand_Index",           "demand_index")
        buyers         = get_int("buyers")
        if buyers is None: buyers = int(ds_medians.get("Number_of_Buyers", 50))
        min_price      = fv("Min_Price",              "min_price")
        max_price      = fv("Max_Price",              "max_price")
        modal_price    = fv("Modal_Price",            "modal_price")
        fuel_price     = fv("Fuel_Price",             "fuel_price")
        transport_cost = fv("Transportation_Cost",    "transport_cost")
        inflation      = fv("Inflation_Rate",         "inflation")

        input_data = {
            "Crop_Name"                : crop_name,
            "State"                    : state,
            "District"                 : district,
            "Market_Name"              : market,
            "Season"                   : season,
            "Previous_Day_Price"       : p_day,
            "Market_Arrival_Quantity"  : arrival,
            "Rainfall"                 : rainfall,
            "Temperature_Max"          : temp_max,
            "Temperature_Min"          : temp_min,
            "Humidity"                 : humidity,
            "Festival_Flag"            : festival_flag,
            "Previous_7_Day_Avg_Price" : p_7,
            "Previous_30_Day_Avg_Price": p_30,
            "Price_Trend"              : price_trend,
            "Production_Quantity"      : production,
            "Yield_Per_Hectare"        : yield_ha,
            "Stock_Available"          : stock,
            "Soil_Moisture"            : soil_moisture,
            "Demand_Index"             : demand_index,
            "Number_of_Buyers"         : buyers,
            "Min_Price"                : min_price,
            "Max_Price"                : max_price,
            "Modal_Price"              : modal_price,
            "Fuel_Price"               : fuel_price,
            "Transportation_Cost"      : transport_cost,
            "Inflation_Rate"           : inflation,
            "Harvest_Season_Flag"      : harvest_flag,
            "Week_Number"              : week_no,
            "Day"                      : day,
            "Month"                    : month,
            "Year"                     : year,
        }

        # ── ML Prediction ─────────────────────────────────────────
        df_pred = pd.DataFrame([input_data])

        for col, le in encoders.items():
            if col in df_pred.columns:
                raw_val = str(df_pred[col].values[0]).strip()
                known   = list(le.classes_)
                if raw_val not in known:
                    print(f"Warning: '{raw_val}' not in '{col}' encoder. Using fallback.")
                    raw_val = known[0]
                    df_pred[col] = raw_val
                df_pred[col] = le.transform(df_pred[col].astype(str))

        for col in model_columns:
            if col not in df_pred.columns:
                df_pred[col] = 0

        df_pred = df_pred[model_columns]
        df_pred = df_pred.apply(pd.to_numeric, errors='coerce').fillna(0)

        ml_price = model.predict(df_pred)[0]
        ml_price = max(0.0, round(float(ml_price), 2))
        print(f"ML Prediction: ₹{ml_price}")

        # ── Gemini Cross-Verification ─────────────────────────────
        gemini_analysis  = None
        gemini_min       = None
        gemini_max       = None
        gemini_confidence= None
        gemini_factors   = []
        gemini_verdict   = None
        final_price      = ml_price

        try:
            if GOOGLE_API_KEY:
                # Build a rich prompt with all 31 features
                trend_emoji = "📈" if price_trend == "Increasing" else ("📉" if price_trend == "Decreasing" else "➡️")
                festival_txt = "Yes (Festival period – demand spike expected)" if festival_flag else "No"
                harvest_txt  = "Yes (Harvest season – supply increase expected)" if harvest_flag else "No"

                gemini_prompt = f"""
You are an expert Indian agricultural market analyst with deep knowledge of India's crop price dynamics.

A farmer needs an accurate price prediction for their crop. An ML model predicted ₹{ml_price:.2f}/quintal.
Cross-verify this prediction using your knowledge of current Indian market trends and provide a complete analysis.

══════════════════════════════════════════════════════════
CROP & MARKET DETAILS
══════════════════════════════════════════════════════════
• Crop          : {crop_name}
• State         : {state}
• District      : {district}
• Market/Mandi  : {market}
• Season        : {season}
• Date          : {date_str}

══════════════════════════════════════════════════════════
PRICE HISTORY (All in ₹/quintal)
══════════════════════════════════════════════════════════
• Previous Day Price  : ₹{p_day:.2f}
• 7-Day Average       : ₹{p_7:.2f}
• 30-Day Average      : ₹{p_30:.2f}
• Auto-Detected Trend : {price_trend} {trend_emoji}
• Min Price (market)  : ₹{min_price:.2f}
• Max Price (market)  : ₹{max_price:.2f}
• Modal Price (market): ₹{modal_price:.2f}

══════════════════════════════════════════════════════════
SUPPLY & PRODUCTION FACTORS
══════════════════════════════════════════════════════════
• Market Arrival Qty  : {arrival:.0f} tonnes
• Production Qty      : {production:.0f} tonnes
• Yield/Hectare       : {yield_ha:.2f} tonnes/ha
• Stock Available     : {stock:.0f} tonnes
• Number of Buyers    : {buyers}
• Demand Index        : {demand_index:.1f}

══════════════════════════════════════════════════════════
WEATHER & ENVIRONMENTAL
══════════════════════════════════════════════════════════
• Rainfall        : {rainfall:.1f} mm
• Max Temperature : {temp_max:.1f}°C
• Min Temperature : {temp_min:.1f}°C
• Humidity        : {humidity:.1f}%
• Soil Moisture   : {soil_moisture:.1f}%

══════════════════════════════════════════════════════════
ECONOMIC FACTORS
══════════════════════════════════════════════════════════
• Fuel Price         : ₹{fuel_price:.2f}/L
• Transportation Cost: ₹{transport_cost:.2f}/tonne
• Inflation Rate     : {inflation:.2f}%
• Festival Period    : {festival_txt}
• Harvest Season     : {harvest_txt}
• Week of Year       : Week {week_no}, Month {month}

══════════════════════════════════════════════════════════
ML MODEL PREDICTION: ₹{ml_price:.2f} per quintal
══════════════════════════════════════════════════════════

Based on ALL the above factors and your knowledge of Indian agricultural markets, provide:

1. MARKET_ANALYSIS: 2-3 sentences explaining current market conditions for {crop_name} in {state}, accounting for the season, supply-demand, weather, and economic factors.

2. PRICE_RANGE: Your estimated realistic price range in format "₹MIN–₹MAX" (e.g., "₹2050–₹2200"). This should be specific, based on the data.

3. ML_VERDICT: Is the ML prediction of ₹{ml_price:.2f} ACCURATE, SLIGHTLY_HIGH, SLIGHTLY_LOW, or SIGNIFICANTLY_DIFFERENT? Give one word.

4. ADJUSTED_PRICE: Your best single adjusted price estimate in ₹ (number only, no text).

5. KEY_FACTORS: List 3 most important factors affecting this price right now (bullet points starting with •).

6. CONFIDENCE: Your confidence level: HIGH / MEDIUM / LOW, with one-line reason.

Format your response EXACTLY like this:
MARKET_ANALYSIS: [your analysis]
PRICE_RANGE: [₹min–₹max]
ML_VERDICT: [ACCURATE/SLIGHTLY_HIGH/SLIGHTLY_LOW/SIGNIFICANTLY_DIFFERENT]
ADJUSTED_PRICE: [number]
KEY_FACTORS:
• [factor 1]
• [factor 2]
• [factor 3]
CONFIDENCE: [HIGH/MEDIUM/LOW] — [reason]
"""
                print("Calling Gemini API for market cross-verification...")
                price_verify_model = genai.GenerativeModel('gemini-3-flash-preview')
                gemini_response = price_verify_model.generate_content(gemini_prompt)
                raw_text = gemini_response.text.strip()
                print("Gemini response:", raw_text[:300])

                # ── Parse Gemini response ─────────────────────────
                import re

                def extract_field(label, text):
                    pattern = rf'{label}:\s*(.+?)(?=\n[A-Z_]+:|$)'
                    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                    return match.group(1).strip() if match else None

                gemini_analysis   = extract_field("MARKET_ANALYSIS", raw_text)
                price_range_raw   = extract_field("PRICE_RANGE", raw_text)
                gemini_verdict    = extract_field("ML_VERDICT", raw_text)
                adjusted_raw      = extract_field("ADJUSTED_PRICE", raw_text)
                confidence_raw    = extract_field("CONFIDENCE", raw_text)

                # Parse price range  ₹2050–₹2200
                if price_range_raw:
                    nums = re.findall(r'[\d,]+\.?\d*', price_range_raw.replace(',', ''))
                    if len(nums) >= 2:
                        try:
                            gemini_min = round(float(nums[0]))
                            gemini_max = round(float(nums[1]))
                        except: pass

                # Parse adjusted price
                if adjusted_raw:
                    adj_nums = re.findall(r'[\d,]+\.?\d*', adjusted_raw.replace(',', ''))
                    if adj_nums:
                        try:
                            adj = round(float(adj_nums[0]), 2)
                            # Blend: Gemini-adjusted 40%, ML 60%
                            final_price = round(0.6 * ml_price + 0.4 * adj, 2)
                            print(f"Adjusted price: ML=₹{ml_price} | Gemini=₹{adj} | Blended=₹{final_price}")
                        except: pass

                # Parse confidence
                if confidence_raw:
                    conf_upper = confidence_raw.upper()
                    if "HIGH" in conf_upper: gemini_confidence = "HIGH"
                    elif "MEDIUM" in conf_upper: gemini_confidence = "MEDIUM"
                    elif "LOW" in conf_upper: gemini_confidence = "LOW"

                # Parse key factors
                factor_lines = re.findall(r'•\s*(.+)', raw_text)
                gemini_factors = [f.strip() for f in factor_lines[:4]]

                if not gemini_analysis:
                    gemini_analysis = raw_text[:400]  # fallback

        except ResourceExhausted:
            print("Gemini quota exceeded — proceeding with ML prediction only.")
        except Exception as ge:
            print(f"Gemini call failed: {ge}")

        print(f"Final prediction: ₹{final_price}")

        return render_template(
            "PricePrediction.html",
            prediction      = final_price,
            ml_price        = ml_price,
            gemini_analysis = gemini_analysis,
            gemini_min      = gemini_min,
            gemini_max      = gemini_max,
            gemini_confidence = gemini_confidence,
            gemini_factors  = gemini_factors,
            gemini_verdict  = gemini_verdict,
            crop_name       = crop_name,
            state           = state,
            season          = season,
            price_trend     = price_trend,
        )

    except Exception as e:
        print(f"PREDICTION FAILED: {e}")
        traceback.print_exc()
        return render_template("PricePrediction.html", error=f"Prediction error: {str(e)}")















@app.route('/chat', methods=['POST'])
def chat():
    if not GOOGLE_API_KEY:
        return jsonify({'error': 'API key not configured'}), 500

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # System prompt to restrict to farming
    system_prompt = """
    You are a helpful farming assistant. 
    Your goal is to assist users with farming-related questions, 
    such as crop cultivation, pest control, soil health, irrigation, and machinery.
    
    If a user asks a question that is NOT related to farming (e.g., general knowledge, politics, entertainment, coding unrelated to farming), 
    you must politely refuse to answer and guide them back to farming topics.
    
    Example of handling unrelated questions:
    User: "Who is the president of the USA?"
    Assistant: "I am a farming assistant and can only answer questions related to agriculture and farming activities. How can I help you with your farm today?"
    
    Now, answer the following user question:
    """
    
    full_prompt = f"{system_prompt}\nUser: {user_message}\nAssistant:"

    try:
        # Using chat_model initialized with legacy SDK
        response = chat_model.generate_content(full_prompt)
        return jsonify({'response': response.text})
    except Exception as e:
        # Better error handling for API errors
        print(f"Error generating content: {e}")
        return jsonify({'error': str(e)}), 500







if __name__ == '__main__':
    # Load ensemble models on startup
    if load_ensemble_models():
        print("\n[OK] All ensemble models loaded successfully!")
    else:
        print("\n[ERROR] Some ensemble models failed to load.")
    
    # Initialize DB (optional, but keeping it for safety)
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', role='admin')
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("Admin user created.")

    app.run(debug=True, host='0.0.0.0', port=5001)
