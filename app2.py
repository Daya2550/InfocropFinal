import os
from flask import Flask, redirect, render_template, request , jsonify, send_file
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import cv2
import joblib



from datetime import datetime
import io
from collections import Counter
from crop_database import get_crop_info, analyze_input_fit
from pdf_generator import CropReportGenerator









disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model_path = "plant_disease_model_1_latest.pt"

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"\n{'!'*60}")
    print(f"CRITICAL ERROR: Model file '{model_path}' NOT FOUND!")
    print(f"Please place the model file in: {os.getcwd()}")
    print(f"{'!'*60}\n")

def prediction(image_path):
    


    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__, static_url_path='/static')
# @app.route('/')
# def home_page():
#     return render_template('home.html')

    
@app.route('/')
def index():
    return render_template('index.html')
    

@app.route('/home')
def home():
    return render_template('home.html')
    

@app.route('/contact')  
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('leaf.html')


# Routes for each crop
@app.route('/barley')
def barley():
    return render_template('barley.html')

@app.route('/maize')
def maize():
    return render_template('maize.html')

@app.route('/rice')
def rice():
    return render_template('rice.html')

@app.route('/soybean')
def soybean():
    return render_template('soybean.html')

@app.route('/cotton')
def cotton():
    return render_template('cotton.html')

@app.route('/sugarcane')
def sugarcane():
    return render_template('sugarcane.html')

# Define routes for each fruits
@app.route('/grapes')
def grapes():
    return render_template('grapes.html')

@app.route('/pomegranate')
def pomegranate():
    return render_template('pomegranate.html')

@app.route('/banana')
def banana():
    return render_template('banana.html')

@app.route('/mango')
def mango():
    return render_template('mango.html')

@app.route('/guava')
def guava():
    return render_template('guava.html')

@app.route('/watermelon')
def watermelon():
    return render_template('watermelon.html')

@app.route('/rose')
def rose():
    return render_template('rose.html')

@app.route('/marigold')
def marigold():
    return render_template('marigold.html')

@app.route('/jasmine')
def jasmine():
    return render_template('jasmine.html')

@app.route('/brinjal')
def brinjal():
    return render_template('brinjal.html')

@app.route('/potato')
def potato():
    return render_template('potato.html')

@app.route('/chilli')
def chilli():
    return render_template('chilli.html')

@app.route('/tomato')
def tomato():
    return render_template('tomato.html')

@app.route('/spinach')
def spinach():
    return render_template('spinach.html')

@app.route('/carrot')
def carrot():
    return render_template('carrot.html')


# Define routes for each crop
@app.route('/aloevera')
def aloevera():
    return render_template('aloevera.html')

@app.route('/amla')
def amla():
    return render_template('amla.html')

@app.route('/ashwagandha')
def ashwagandha():
    return render_template('ashwagandha.html')

@app.route('/pages-faq')
def faq():
    return render_template('faq.html')

@app.route('/pages-register')
def register():
    return render_template('register.html')  
 
@app.route('/mobile-device')     
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                        supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))













pdf_gen = CropReportGenerator()

#Ensemble model paths
MODEL_PATHS = {
    'random_forest': 'random_forest_model.pkl',
    'gradient_boosting': 'gradient_boosting_model.pkl',
    'decision_tree': 'decision_tree_model.pkl'
}

# Global models dictionary
models = {}

def load_ensemble_models():
    """Load all trained ML models for ensemble prediction."""
    global models
    success_count = 0
    
    print("\n" + "=" * 80)
    print("LOADING ENSEMBLE MODELS")
    print("=" * 80)
    
    for model_name, model_path in MODEL_PATHS.items():
        try:
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
                print(f"[OK] Loaded {model_name.replace('_', ' ').title()}")
                success_count += 1
            else:
                print(f"[ERROR] Model file not found: {model_path}")
        except Exception as e:
            print(f"[ERROR] Error loading {model_name}: {e}")
    
    print(f"\n[OK] Successfully loaded {success_count}/{len(MODEL_PATHS)} models")
    print("=" * 80)
    
    return success_count == len(MODEL_PATHS)

# Input validation ranges with stricter rules
VALIDATION_RANGES = {
    'nitrogen': (0, 150),
    'phosphorus': (0, 150),
    'potassium': (0, 150),
    'temperature': (0, 60),
    'humidity': (0, 100),
    'ph': (0, 14),
    'rainfall': (0, 500)
}

# Stricter validation rules - logical consistency checks
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

def validate_input(data):
    """
    Validate input parameters with stricter rules.
    
    Returns:
        tuple: (is_valid, error_message, validated_data, warnings)
    """
    param_names = {
        'nitrogen': 'Nitrogen',
        'phosphorus': 'Phosphorus',
        'potassium': 'Potassium',
        'temperature': 'Temperature',
        'humidity': 'Humidity',
        'ph': 'pH',
        'rainfall': 'Rainfall'
    }
    
    validated = {}
    warnings_list = []
    
    # Basic validation
    for param, (min_val, max_val) in VALIDATION_RANGES.items():
        try:
            value = float(data.get(param, ''))
            
            # Check range
            if not min_val <= value <= max_val:
                return False, f"Invalid {param_names[param]}: must be between {min_val} and {max_val}", None, []
            
            validated[param] = value
        except (ValueError, TypeError):
            return False, f"Invalid {param_names[param]}: must be a valid number", None, []
    
    # Strict rules validation
    for rule_name, rule in STRICT_RULES.items():
        if rule['condition'](validated):
            warnings_list.append(rule['warning'])
    
    return True, "Valid", validated, warnings_list

def predict_ensemble(input_data, user_inputs_dict):
    """
    Make predictions using all models and return ensemble result with explanations.
    Uses a hybrid voting strategy (Hard + Soft Voting) for maximum reliability.
    """
    predictions = {}
    probabilities = {}
    model_probs = {} # Full probability distribution for each model
    
    # Get prediction from each model
    for model_name, model in models.items():
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)
        confidence = float(np.max(proba)) * 100
        
        predictions[model_name] = {
            'crop': pred,
            'confidence': round(confidence, 2)
        }
        
        # Store probability and classes
        model_probs[model_name] = proba[0]
        classes = model.classes_
    
    # --- ENHANCED VOTING LOGIC ---
    crops = [p['crop'] for p in predictions.values()]
    vote_counts = Counter(crops)
    most_common_items = vote_counts.most_common()
    
    most_common_crop, vote_count = most_common_items[0]
    is_tie_breaker = False
    
    # If all models disagree (tie), use Soft Voting (sum of probabilities)
    if vote_count == 1 and len(crops) > 1:
        is_tie_breaker = True
        soft_vote_scores = {}
        
        for i, crop_label in enumerate(classes):
            total_prob = sum(model_probs[m][i] for m in models.keys())
            soft_vote_scores[crop_label] = total_prob
            
        # Select crop with highest cumulative probability score
        most_common_crop = max(soft_vote_scores, key=soft_vote_scores.get)
        vote_count = "Tie-Breaker"
    
    # Calculate agreement percentage (for majority cases)
    if isinstance(vote_count, int):
        agreement = (vote_count / len(crops)) * 100
    else:
        agreement = 0 # Indicative of no majority
    
    # Calculate average confidence for the winning crop (Soft Confidence)
    avg_confidence = np.mean([p['confidence'] for p in predictions.values()])
    
    # Get feature importance from Random Forest (most interpretable)
    feature_names = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    feature_importance = {}
    
    if 'random_forest' in models:
        rf_model = models['random_forest']
        importances = rf_model.feature_importances_
        
        # Get top 3 most important features
        feature_importance = {
            feature_names[i]: round(float(importances[i]) * 100, 2)
            for i in range(len(feature_names))
        }
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:3]
    else:
        top_features = []
    
    # Get crop information
    crop_info = get_crop_info(most_common_crop)
    
    # Analyze input fit
    input_analysis = analyze_input_fit(most_common_crop, user_inputs_dict)
    
    # Generate reasoning
    reasoning = generate_reasoning(most_common_crop, top_features, input_analysis, user_inputs_dict)
    
    return {
        'final_crop': most_common_crop,
        'confidence': round(avg_confidence, 2),
        'agreement': round(agreement, 2) if isinstance(agreement, (int, float)) else agreement,
        'vote_count': f"{vote_count}/{len(crops)}" if isinstance(vote_count, int) else vote_count,
        'individual_predictions': predictions,
        'unanimous': vote_count == len(crops) if isinstance(vote_count, int) else False,
        'is_tie_breaker': is_tie_breaker,
        'feature_importance': dict(sorted_features[:5]),  # Top 5 features
        'top_influencing_factors': [{'factor': f[0], 'importance': f[1]} for f in top_features],
        'crop_info': crop_info,
        'input_analysis': input_analysis,
        'reasoning': reasoning
    }

def generate_reasoning(crop_name, top_features, input_analysis, user_inputs):
    """
    Generate human-readable reasoning for the crop recommendation.
    
    Args:
        crop_name: Recommended crop name
        top_features: List of (feature_name, importance) tuples
        input_analysis: Analysis of how inputs match crop requirements
        user_inputs: User's input values
        
    Returns:
        list: List of reasoning points
    """
    reasoning = []
    
    # Main recommendation
    reasoning.append(f"{crop_name.title()} is recommended based on your soil and weather conditions.")
    
    # Feature influence
    if top_features:
        top_factor = top_features[0]
        reasoning.append(
            f"The most influential factor is {top_factor[0]} ({top_factor[1]}% importance), "
            f"with your value of {user_inputs.get(top_factor[0].lower(), 'N/A')}."
        )
    
    # Input match
    if input_analysis['match_percentage'] != 'N/A':
        match_pct = input_analysis['match_percentage']
        if match_pct >= 80:
            reasoning.append(f"Your conditions match {match_pct}% with optimal {crop_name} requirements - Excellent fit!")
        elif match_pct >= 60:
            reasoning.append(f"Your conditions match {match_pct}% with optimal {crop_name} requirements - Good fit.")
        else:
            reasoning.append(f"Your conditions match {match_pct}% with optimal {crop_name} requirements.")
    
    # Matching factors
    if input_analysis['matching_factors']:
        reasoning.append("Favorable conditions: " + ", ".join(input_analysis['matching_factors'][:2]))
    
    return reasoning




















@app.route('/CropRec')
def index1():
    """Render the main page."""
    return render_template('CropRec.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """API endpoint for crop recommendation using ensemble."""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input with strict rules
        is_valid, message, validated_data, warnings = validate_input(data)
        
        if not is_valid:
            return jsonify({
                'success': False,
                'error': message
            }), 400
        
        # Check if models are loaded
        if not models:
            return jsonify({
                'success': False,
                'error': 'Models not loaded. Please contact administrator.'
            }), 500
        
        # Prepare input for prediction
        input_array = np.array([[
            validated_data['nitrogen'],
            validated_data['phosphorus'],
            validated_data['potassium'],
            validated_data['temperature'],
            validated_data['humidity'],
            validated_data['ph'],
            validated_data['rainfall']
        ]])
        
        # Make ensemble prediction with explanations
        result = predict_ensemble(input_array, validated_data)
        
        return jsonify({
            'success': True,
            'crop': result['final_crop'],
            'confidence': result['confidence'],
            'agreement': result['agreement'],
            'vote_count': result['vote_count'],
            'unanimous': result['unanimous'],
            'is_tie_breaker': result['is_tie_breaker'],
            'individual_predictions': result['individual_predictions'],
            'warnings': warnings,
            'feature_importance': result['feature_importance'],
            'top_influencing_factors': result['top_influencing_factors'],
            'crop_info': result['crop_info'],
            'input_analysis': result['input_analysis'],
            'reasoning': result['reasoning']
        })
        
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing your request.'
        }), 500

@app.route('/api/info')
def info():
    """API endpoint for system information."""
    return jsonify({
        'models_loaded': len(models),
        'model_names': list(models.keys()),
        'validation_ranges': VALIDATION_RANGES,
        'strict_rules': len(STRICT_RULES)
    })

@app.route('/api/download_report', methods=['POST'])
def download_report():
    """API endpoint to generate and download a PDF report."""
    try:
        data = request.get_json()
        prediction_result = data.get('prediction_data')
        user_inputs = data.get('user_inputs')
        
        if not prediction_result or not user_inputs:
            return jsonify({'success': False, 'error': 'Missing required data'}), 400
            
        # Generate PDF
        pdf_buffer = pdf_gen.generate_report(prediction_result, user_inputs)
        
        # Prepare filename
        crop_name = prediction_result.get('crop', 'recommendation').lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crop_report_{crop_name}_{timestamp}.pdf"
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"[ERROR] PDF generation error: {e}")
        return jsonify({'success': False, 'error': 'Failed to generate PDF report'}), 500





if __name__ == '__main__':
 # Load ensemble models on startup
    if load_ensemble_models():
        print("\n[OK] All models loaded successfully!")
    else:
        print("\n[ERROR] Some models failed to load. Application may not work correctly.")
    
    # Run the Flask app
    print("\n" + "=" * 80)
    print("CROP RECOMMENDATION WEB APPLICATION - ENSEMBLE VERSION")
    print("=" * 80)
    print("\nFeatures:")
    print("  - Multi-model ensemble prediction")
    print("  - 3 ML models working together")
    print("  - Stricter validation rules")
    print("  - Logical consistency checks")
    print("\nStarting server...")
    print("Access the application at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)





