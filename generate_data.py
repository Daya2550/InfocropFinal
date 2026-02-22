import pandas as pd
import random
from datetime import datetime, timedelta

# Configuration
num_records = 10000 
crops = ['Wheat', 'Rice', 'Onion', 'Potato', 'Tomato']

# Expanded Real-World Location Data (State -> District -> Markets)
locations = {
    'Punjab': {
        'Ludhiana': ['Khanna', 'Sahnewal', 'Mullanpur'],
        'Amritsar': ['Rayya', 'Gehri Mandi', 'Majitha'],
        'Patiala': ['Rajpura', 'Patiala_Mandi', 'Samana'],
        'Jalandhar': ['Jalandhar_City', 'Adampur', 'Nakodar']
    },
    'Maharashtra': {
        'Pune': ['Pune_Market', 'Manchar', 'Khed'],
        'Nashik': ['Lasalgaon', 'Pimpalgaon', 'Yeola'],
        'Nagpur': ['Nagpur_Mandi', 'Kalmeshwar', 'Katol'],
        'Ahmednagar': ['Ahmednagar_Mandi', 'Rahata', 'Kopargaon']
    },
    'Uttar Pradesh': {
        'Agra': ['Agra_Mandi', 'Fatehabad', 'Kheragarh'],
        'Kanpur': ['Kanpur_Mandi', 'Chaubepur', 'Bilhaur'],
        'Lucknow': ['Lucknow_Mandi', 'Mohanlalganj', 'Malihabad'],
        'Meerut': ['Meerut_Mandi', 'Mawana', 'Sardhana']
    },
    'Haryana': {
        'Karnal': ['Karnal_Mandi', 'Gharaunda', 'Taraori'],
        'Kurukshetra': ['Thanesar', 'Pehowa', 'Shahabad'],
        'Ambala': ['Ambala_City', 'Ambala_Cantt', 'Naraingarh']
    },
    'Madhya Pradesh': {
        'Indore': ['Indore_Mandi', 'Mhow', 'Depalpur'],
        'Bhopal': ['Bhopal_Mandi', 'Berasia'],
        'Ujjain': ['Ujjain_Mandi', 'Barnagar', 'Mahidpur']
    }
}

seasons = ['Rabi', 'Kharif', 'Zaid']
start_date = datetime(2021, 1, 1) # Last ~4-5 years
end_date = datetime.now()
days_range = (end_date - start_date).days

# Base Prices (approximate ₹/Quintal)
base_prices = {
    'Wheat': 2275,   # MSP approx
    'Rice': 2203,    # MSP approx
    'Onion': 2500,   # Volatile
    'Potato': 1200,
    'Tomato': 3000   # Volatile
}

data = []

print(f"Generating {num_records} records...")

for _ in range(num_records):
    # Random Location
    state = random.choice(list(locations.keys()))
    district = random.choice(list(locations[state].keys()))
    market = random.choice(locations[state][district])
    
    # Random Crop
    crop = random.choice(crops)
    
    # Random Date
    random_days = random.randint(0, days_range)
    date = start_date + timedelta(days=random_days)
    date_str = date.strftime('%Y-%m-%d')
    month = date.month
    
    # Determine Season based on Month (Rough approximation for India)
    if month in [11, 12, 1, 2, 3, 4]:
        season = 'Rabi'
    elif month in [6, 7, 8, 9, 10]:
        season = 'Kharif'
    else:
        season = 'Zaid' # May, June (Summer)
        
    # Weather (Random but somewhat correlated with season/state could be better, but simple ranges for now)
    # Monsoon (June-Sept) -> High Rain, High Humidity
    is_monsoon = month in [6, 7, 8, 9]
    is_winter = month in [11, 12, 1, 2]
    is_summer = month in [3, 4, 5]
    
    if is_monsoon:
        rainfall = round(random.uniform(100, 300), 1)
        humidity = round(random.uniform(70, 95), 1)
        temp_max = round(random.uniform(30, 35), 1)
        temp_min = round(random.uniform(24, 28), 1)
    elif is_winter:
        rainfall = round(random.uniform(0, 20), 1)
        humidity = round(random.uniform(40, 70), 1)
        temp_max = round(random.uniform(15, 25), 1) # Cooler in North
        temp_min = round(random.uniform(5, 12), 1)
    else: # Summer
        rainfall = round(random.uniform(0, 50), 1)
        humidity = round(random.uniform(30, 60), 1)
        temp_max = round(random.uniform(35, 45), 1)
        temp_min = round(random.uniform(20, 30), 1)
        
    # State specific tweaks (e.g. Maha/South is warmer in winter)
    if state in ['Maharashtra', 'Madhya Pradesh'] and is_winter:
        temp_max += 5
        temp_min += 5

    # Market Factors
    festival = 1 if random.random() < 0.1 else 0 # 10% chance of festival
    arrival = round(random.uniform(50, 5000), 1)
    
    # Price Logic
    current_base = base_prices[crop]
    
    # Seasonality price impact
    # Off-season prices are higher
    season_factor = 0
    if crop == 'Wheat' and season != 'Rabi': season_factor = 200
    if crop == 'Rice' and season != 'Kharif': season_factor = 200
    if crop == 'Onion' and month in [10, 11, 12]: season_factor = 1000 # Late kharif harvest shortage often spikes prices
    
    # Volatility
    price_noise = random.uniform(-0.15, 0.15) * current_base # +/- 15% variation
    
    festival_bump = 0
    if festival:
        festival_bump = random.uniform(200, 500)
        
    arrival_impact = 0
    if arrival > 4000: # Glut
        arrival_impact = -150
    elif arrival < 500: # Shortage
        arrival_impact = 150
        
    estimated_price = current_base + season_factor + price_noise + festival_bump + arrival_impact
    
    # Ensure realistic
    estimated_price = max(500, estimated_price)
    
    # Prev price
    prev_price = round(estimated_price + random.uniform(-100, 100), 2)
    final_price = round(estimated_price, 2)
    
    record = {
        "Date": date_str,
        "Crop_Name": crop,
        "State": state,
        "District": district,
        "Market_Name": market,
        "Season": season,
        "Previous_Day_Price": prev_price,
        "Market_Arrival_Quantity": arrival,
        "Rainfall": rainfall,
        "Temperature_Max": temp_max,
        "Temperature_Min": temp_min,
        "Humidity": humidity,
        "Festival_Flag": festival,
        "Price": final_price
    }
    data.append(record)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("crop_price_dataset.csv", index=False)
print(f"Successfully generated {num_records} records in crop_price_dataset.csv")
