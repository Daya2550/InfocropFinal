"""
Crop Knowledge Database
Contains optimal growing conditions and requirements for each crop.
"""

CROP_REQUIREMENTS = {
    'rice': {
        'name': 'Rice',
        'optimal_ranges': {
            'nitrogen': (80, 100),
            'phosphorus': (35, 50),
            'potassium': (35, 50),
            'temperature': (20, 27),
            'humidity': (75, 90),
            'ph': (6.0, 7.0),
            'rainfall': (150, 300)
        },
        'description': 'Rice thrives in warm, humid climates with consistent water availability.',
        'growing_conditions': [
            'Requires flooded fields or consistent irrigation',
            'Warm temperatures throughout growing season',
            'High humidity levels preferred',
            'Neutral to slightly acidic soil'
        ],
        'season': 'Monsoon/Rainy season',
        'water_requirement': 'High',
        'climate': 'Tropical and subtropical'
    },
    'maize': {
        'name': 'Maize (Corn)',
        'optimal_ranges': {
            'nitrogen': (70, 100),
            'phosphorus': (40, 60),
            'potassium': (15, 25),
            'temperature': (18, 27),
            'humidity': (55, 75),
            'ph': (5.5, 7.5),
            'rainfall': (50, 110)
        },
        'description': 'Maize grows well in moderate climates with adequate drainage.',
        'growing_conditions': [
            'Well-drained soil essential',
            'Moderate temperature range',
            'Lower potassium requirement',
            'Tolerates slight acidity'
        ],
        'season': 'Spring to summer',
        'water_requirement': 'Moderate',
        'climate': 'Temperate to warm'
    },
    'chickpea': {
        'name': 'Chickpea',
        'optimal_ranges': {
            'nitrogen': (20, 60),
            'phosphorus': (55, 80),
            'potassium': (75, 85),
            'temperature': (17, 21),
            'humidity': (14, 20),
            'ph': (6.0, 8.5),
            'rainfall': (65, 95)
        },
        'description': 'Chickpeas prefer cooler, drier climates with well-drained soil.',
        'growing_conditions': [
            'Cool season crop',
            'Low humidity requirement',
            'High phosphorus and potassium needs',
            'Drought tolerant once established'
        ],
        'season': 'Winter (Rabi season)',
        'water_requirement': 'Low to moderate',
        'climate': 'Cool and dry'
    },
    'kidneybeans': {
        'name': 'Kidney Beans',
        'optimal_ranges': {
            'nitrogen': (10, 40),
            'phosphorus': (55, 80),
            'potassium': (15, 25),
            'temperature': (15, 24),
            'humidity': (18, 25),
            'ph': (5.5, 7.0),
            'rainfall': (60, 145)
        },
        'description': 'Kidney beans grow best in cool temperatures with moderate moisture.',
        'growing_conditions': [
            'Cool season legume',
            'Low nitrogen needs (fixes own nitrogen)',
            'Requires good drainage',
            'Sensitive to frost'
        ],
        'season': 'Winter to spring',
        'water_requirement': 'Moderate',
        'climate': 'Cool temperate'
    },
    'pigeonpeas': {
        'name': 'Pigeon Peas',
        'optimal_ranges': {
            'nitrogen': (10, 40),
            'phosphorus': (55, 80),
            'potassium': (15, 25),
            'temperature': (20, 35),
            'humidity': (30, 70),
            'ph': (5.0, 7.5),
            'rainfall': (90, 200)
        },
        'description': 'Pigeon peas are drought-resistant and thrive in tropical climates.',
        'growing_conditions': [
            'Drought and heat tolerant',
            'Wide pH tolerance',
            'Low fertilizer requirement',
            'Deep root system'
        ],
        'season': 'Post-monsoon',
        'water_requirement': 'Low',
        'climate': 'Tropical and subtropical'
    },
    'mothbeans': {
        'name': 'Moth Beans',
        'optimal_ranges': {
            'nitrogen': (10, 40),
            'phosphorus': (35, 60),
            'potassium': (15, 25),
            'temperature': (24, 32),
            'humidity': (40, 65),
            'ph': (5.0, 9.0),
            'rainfall': (30, 75)
        },
        'description': 'Moth beans are extremely drought-resistant and heat-tolerant.',
        'growing_conditions': [
            'Highly drought resistant',
            'Thrives in arid conditions',
            'Low water requirement',
            'Sandy soil preferred'
        ],
        'season': 'Summer (Kharif)',
        'water_requirement': 'Very low',
        'climate': 'Hot and dry'
    },
    'mungbean': {
        'name': 'Mung Bean',
        'optimal_ranges': {
            'nitrogen': (10, 40),
            'phosphorus': (35, 60),
            'potassium': (15, 25),
            'temperature': (27, 30),
            'humidity': (80, 90),
            'ph': (6.2, 7.5),
            'rainfall': (36, 60)
        },
        'description': 'Mung beans prefer warm, humid conditions with well-drained soil.',
        'growing_conditions': [
            'Short duration crop',
            'High humidity tolerance',
            'Warm temperatures essential',
            'Good drainage required'
        ],
        'season': 'Summer (Kharif)',
        'water_requirement': 'Moderate',
        'climate': 'Warm and humid'
    },
    'blackgram': {
        'name': 'Black Gram',
        'optimal_ranges': {
            'nitrogen': (20, 60),
            'phosphorus': (55, 80),
            'potassium': (15, 25),
            'temperature': (25, 35),
            'humidity': (60, 70),
            'ph': (6.5, 7.8),
            'rainfall': (60, 75)
        },
        'description': 'Black gram grows well in warm climates with moderate rainfall.',
        'growing_conditions': [
            'Warm season pulse',
            'Moderate water needs',
            'Well-aerated soil preferred',
            'Sensitive to waterlogging'
        ],
        'season': 'Summer (Kharif)',
        'water_requirement': 'Moderate',
        'climate': 'Warm tropical'
    },
    'cotton': {
        'name': 'Cotton',
        'optimal_ranges': {
            'nitrogen': (100, 140),
            'phosphorus': (35, 60),
            'potassium': (35, 60),
            'temperature': (21, 30),
            'humidity': (50, 80),
            'ph': (6.0, 8.0),
            'rainfall': (60, 120)
        },
        'description': 'Cotton requires warm temperatures and long frost-free periods.',
        'growing_conditions': [
            'Long growing season required',
            'High nitrogen needs',
            'Frost-free period essential',
            'Deep well-drained soil'
        ],
        'season': 'Summer (Kharif)',
        'water_requirement': 'High',
        'climate': 'Warm subtropical'
    }
}

def get_crop_info(crop_name):
    """
    Get detailed information about a specific crop.
    
    Args:
        crop_name: Name of the crop (lowercase)
        
    Returns:
        dict: Crop information or default info if crop not in database
    """
    crop_name = crop_name.lower()
    
    if crop_name in CROP_REQUIREMENTS:
        return CROP_REQUIREMENTS[crop_name]
    else:
        # Default info for crops not in database
        return {
            'name': crop_name.title(),
            'description': f'{crop_name.title()} cultivation requires specific soil and weather conditions.',
            'growing_conditions': ['Consult local agricultural experts for specific requirements'],
            'season': 'Varies by region',
            'water_requirement': 'Moderate',
            'climate': 'Varies'
        }

def analyze_input_fit(crop_name, user_inputs):
    """
    Analyze how well user inputs match crop requirements.
    
    Args:
        crop_name: Name of the recommended crop
        user_inputs: Dictionary of user's input values
        
    Returns:
        dict: Analysis of input fit with crop requirements
    """
    crop_info = get_crop_info(crop_name)
    
    if 'optimal_ranges' not in crop_info:
        return {'match_percentage': 'N/A', 'matching_factors': [], 'suggestions': []}
    
    optimal = crop_info['optimal_ranges']
    matching_factors = []
    suggestions = []
    
    param_names = {
        'nitrogen': 'Nitrogen',
        'phosphorus': 'Phosphorus',
        'potassium': 'Potassium',
        'temperature': 'Temperature',
        'humidity': 'Humidity',
        'ph': 'pH',
        'rainfall': 'Rainfall'
    }
    
    for param, value in user_inputs.items():
        if param in optimal:
            min_val, max_val = optimal[param]
            
            if min_val <= value <= max_val:
                matching_factors.append(f"{param_names[param]} is optimal ({value})")
            elif value < min_val:
                diff_percent = ((min_val - value) / min_val) * 100
                if diff_percent < 20:
                    matching_factors.append(f"{param_names[param]} is slightly below optimal ({value})")
                else:
                    suggestions.append(f"Consider increasing {param_names[param]} (current: {value}, optimal: {min_val}-{max_val})")
            elif value > max_val:
                diff_percent = ((value - max_val) / max_val) * 100
                if diff_percent < 20:
                    matching_factors.append(f"{param_names[param]} is slightly above optimal ({value})")
                else:
                    suggestions.append(f"Consider decreasing {param_names[param]} (current: {value}, optimal: {min_val}-{max_val})")
    
    # Calculate match percentage
    total_params = len(optimal)
    exact_matches = sum(1 for param, value in user_inputs.items() 
                       if param in optimal and optimal[param][0] <= value <= optimal[param][1])
    match_percentage = (exact_matches / total_params) * 100 if total_params > 0 else 0
    
    return {
        'match_percentage': round(match_percentage, 1),
        'matching_factors': matching_factors,
        'suggestions': suggestions
    }
