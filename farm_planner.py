"""
Farm Pre-Crop Information System
=================================
Flask Blueprint – 10-stage guided wizard for farmers.
Each stage collects focused inputs, calls Gemini AI with all
accumulated context, and returns actionable real-time guidance.
Final stage generates a downloadable PDF farm plan.
"""

from flask import (
    Blueprint, render_template, request, session,
    redirect, url_for, make_response, jsonify
)
import google.generativeai as genai
import os, json, traceback, uuid, tempfile
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import io

# ── Server-side farm data storage (bypasses 4KB cookie limit) ──
# The full farm plan (including long Gemini responses) is stored as
# a JSON file in the OS temp dir.  Only a small UUID is kept in the
# Flask session cookie.

_FARM_TMPDIR = os.path.join(tempfile.gettempdir(), 'infocrop_farm_plans')
os.makedirs(_FARM_TMPDIR, exist_ok=True)

def _farm_path(farm_id: str) -> str:
    # Sanitize to prevent path traversal
    safe_id = farm_id.replace('/', '').replace('\\', '')[:64]
    return os.path.join(_FARM_TMPDIR, f'{safe_id}.json')

def get_farm_data() -> dict:
    """Load farm plan from server-side JSON file."""
    farm_id = session.get('farm_id')
    if not farm_id:
        return {}
    try:
        path = _farm_path(farm_id)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f'[FarmPlanner] Error loading farm data: {e}')
    return {}

def save_farm_data(data: dict):
    """Persist farm plan to server-side JSON file, create ID if needed."""
    farm_id = session.get('farm_id')
    if not farm_id:
        farm_id = uuid.uuid4().hex
        session['farm_id'] = farm_id
        session.modified = True
    try:
        path = _farm_path(farm_id)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f'[FarmPlanner] Error saving farm data: {e}')

def clear_farm_data():
    """Delete the server-side file and remove the session key."""
    farm_id = session.pop('farm_id', None)
    if farm_id:
        try:
            path = _farm_path(farm_id)
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f'[FarmPlanner] Error deleting farm data: {e}')
    session.modified = True

farm_bp = Blueprint('farm_planner', __name__, url_prefix='/farm-planner')

# NOTE: genai is configured lazily inside run_gemini_stage() so that
# load_dotenv() in app.py has already run before we read the key.

# ══════════════════════════════════════════════════════════════
#  STAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════
STAGES = [
    {
        "num"   : 1,
        "emoji" : "🌱",
        "title" : "Crop Selection",
        "subtitle": "Tell us about your land and goals",
        "color" : "#2e7d32",
        "fields": [
            {"name": "farmer_name",  "label": "Your Name",           "type": "text",   "required": True,  "placeholder": "e.g. Ramesh Kumar"},
            {"name": "city",         "label": "City / Taluka",        "type": "text",   "required": True,  "placeholder": "e.g. Nashik"},
            {"name": "state",        "label": "State",                "type": "select", "required": True,
             "options": ["Andhra Pradesh","Assam","Bihar","Chhattisgarh","Gujarat","Haryana",
                         "Himachal Pradesh","Jharkhand","Karnataka","Kerala","Madhya Pradesh",
                         "Maharashtra","Manipur","Odisha","Punjab","Rajasthan","Tamil Nadu",
                         "Telangana","Uttar Pradesh","Uttarakhand","West Bengal","Other"]},
            {"name": "land_area",    "label": "Land Area (Acres)",    "type": "number", "required": True,  "placeholder": "e.g. 5", "min": 0.1, "step": 0.1},
            {"name": "soil_type",    "label": "Soil Type",            "type": "select", "required": True,
             "options": ["Select", "Black (Regur)", "Red & Laterite", "Alluvial", "Sandy / Desert", "Loamy", "Clay", "I don't know"]},
            {"name": "season",       "label": "Current Season",       "type": "select", "required": True,
             "options": ["Select", "Kharif (Jun–Oct)", "Rabi (Nov–Mar)", "Zaid (Mar–Jun)"]},
            {"name": "budget",       "label": "Total Budget (₹)",     "type": "number", "required": True,  "placeholder": "e.g. 50000", "min": 1000},
            {"name": "water_source", "label": "Water Source",         "type": "select", "required": True,
             "options": ["Select", "Canal (nehr)", "Borewell", "Rain-fed only", "River / Pond", "Mixed"]},
            {"name": "crop_goal",    "label": "Main Goal",            "type": "select", "required": False,
             "options": ["Maximum profit", "Family food security", "Both profit + food", "Export quality crop"]},
        ],
        "gemini_key": "crop_recommendation",
        "prompt_fn": lambda d: f"""
You are an expert Indian agricultural advisor. A farmer has shared the following details:

👤 Farmer: {d.get('farmer_name','N/A')}
📍 Location: {d.get('city','N/A')}, {d.get('state','N/A')}
🌾 Land: {d.get('land_area','N/A')} acres
🌍 Soil Type: {d.get('soil_type','N/A')}
📅 Season: {d.get('season','N/A')}
💰 Budget: ₹{d.get('budget','N/A')}
💧 Water: {d.get('water_source','N/A')}
🎯 Goal: {d.get('crop_goal','N/A')}
📆 Current Date: {datetime.now().strftime('%B %d, %Y')}

Based on current market trends, Indian government MSP rates 2024-25, local climate suitability, and the farmer's budget, provide:

1. **TOP_CROP**: Best single crop recommendation with 1-line reason
2. **ALTERNATIVES**: 2 alternative crops (comma separated), each with a brief reason
3. **WHY_BEST**: 3 specific reasons why the top crop is ideal for this exact location/season/budget
4. **MARKET_OUTLOOK**: Current market trend for recommended crop in {d.get('state','India')} (3-4 lines, include current MSP if applicable)
5. **SEASON_ADVICE**: Key seasonal timing advice for {d.get('state','N/A')} right now
6. **BUDGET_FIT**: How this crop fits the ₹{d.get('budget','N/A')} budget with rough cost breakdown

Format with clear headings. Be specific to {d.get('state','India')} conditions. Mention real 2024-25 MSP or market prices where applicable.
"""
    },
    {
        "num"   : 2,
        "emoji" : "🌾",
        "title" : "Seed Selection",
        "subtitle": "Choose the right seed variety for your crop",
        "color" : "#558b2f",
        "fields": [
            {"name": "confirmed_crop",  "label": "Confirmed Crop",        "type": "text",   "required": True,  "placeholder": "From AI recommendation or your choice"},
            {"name": "seed_preference", "label": "Seed Preference",       "type": "select", "required": False,
             "options": ["No preference", "Hybrid (high yield)", "Traditional/Desi variety", "Organic certified", "Government supplied"]},
            {"name": "irrigation_avail","label": "Is irrigation available?","type":"select", "required": True,
             "options": ["Yes – full irrigation", "Partial / limited water", "Rain-fed only"]},
            {"name": "prev_crop",       "label": "Last crop grown here",   "type": "text",   "required": False, "placeholder": "e.g. Wheat, or 'None'"},
            {"name": "disease_history", "label": "Any known disease problems in this field?","type":"text","required":False,"placeholder":"e.g. Root rot, leaf blight, or 'None'"},
        ],
        "gemini_key": "seed_advice",
        "prompt_fn": lambda d: f"""
You are an expert Indian seed scientist and agronomist.

FARMER CONTEXT:
- Location: {d.get('city','N/A')}, {d.get('state','N/A')}
- Crop chosen: {d.get('confirmed_crop', d.get('crop_recommendation_TOP_CROP','N/A'))}
- Land: {d.get('land_area','N/A')} acres, Soil: {d.get('soil_type','N/A')}
- Season: {d.get('season','N/A')}
- Irrigation: {d.get('irrigation_avail','N/A')}
- Previous crop: {d.get('prev_crop','N/A')}
- Known diseases: {d.get('disease_history','None')}
- Budget: ₹{d.get('budget','N/A')}
- Date: {datetime.now().strftime('%B %Y')}

Provide detailed seed advice:

1. **BEST_VARIETY**: Top variety name + certification body
2. **SEED_QUANTITY**: Seeds needed for {d.get('land_area','N/A')} acres (kg or packets)
3. **SEED_COST**: Approximate seed cost in ₹
4. **WHERE_TO_BUY**: Best sources (government, ICAR stations, private brands specific to {d.get('state','N/A')})
5. **GERMINATION_TIPS**: 3 specific tips to maximize germination rate in {d.get('state','N/A')} climate
6. **DISEASE_RESISTANT**: Is this variety resistant to {d.get('disease_history','common diseases')}?
7. **ROTATION_BENEFIT**: Benefit of growing after {d.get('prev_crop','previous crop')}

Be specific to {d.get('state','India')} and the current {d.get('season','season')}.
"""
    },
    {
        "num"   : 3,
        "emoji" : "🌍",
        "title" : "Soil Health",
        "subtitle": "Understand your soil and get a fertilizer plan",
        "color" : "#6d4c41",
        "fields": [
            {"name": "ph_level",    "label": "Soil pH (if known)",     "type": "number","required": False, "placeholder": "e.g. 6.5", "min": 0, "max": 14, "step": 0.1},
            {"name": "nitrogen",    "label": "Nitrogen level (N)",      "type": "select","required": False,
             "options": ["Unknown", "Very Low", "Low", "Medium", "High"]},
            {"name": "phosphorus",  "label": "Phosphorus level (P)",    "type": "select","required": False,
             "options": ["Unknown", "Very Low", "Low", "Medium", "High"]},
            {"name": "potassium",   "label": "Potassium level (K)",     "type": "select","required": False,
             "options": ["Unknown", "Very Low", "Low", "Medium", "High"]},
            {"name": "organic_matter","label": "Organic matter / compost used?","type":"select","required":False,
             "options": ["No", "Yes – FYM / cow dung", "Yes – vermicompost", "Yes – green manure"]},
            {"name": "last_fertilizer","label": "Last fertilizer used", "type": "text", "required": False, "placeholder": "e.g. DAP + Urea, or None"},
        ],
        "gemini_key": "soil_plan",
        "prompt_fn": lambda d: f"""
You are a certified soil scientist advising Indian farmers.

FARM DETAILS:
- Location: {d.get('city','N/A')}, {d.get('state','N/A')}
- Crop: {d.get('confirmed_crop', d.get('crop_recommendation_TOP_CROP','N/A'))}
- Land: {d.get('land_area','N/A')} acres
- Soil type: {d.get('soil_type','N/A')}
- Soil pH: {d.get('ph_level','Unknown')}
- Nitrogen: {d.get('nitrogen','Unknown')} | Phosphorus: {d.get('phosphorus','Unknown')} | Potassium: {d.get('potassium','Unknown')}
- Organic matter: {d.get('organic_matter','No')}
- Last fertilizer: {d.get('last_fertilizer','Unknown')}
- Budget: ₹{d.get('budget','N/A')}

Provide a complete soil and fertilizer management plan:

1. **SOIL_ASSESSMENT**: Current soil health summary with concerns
2. **PH_ACTION**: pH correction needed? How? (lime / sulphur quantities per acre)
3. **FERTILIZER_SCHEDULE**: Stage-wise fertilizer plan (basal, top-dress) with exact kg/acre, specific product names available in {d.get('state','India')}
4. **ORGANIC_BOOST**: Organic additions recommended and their benefits
5. **MICRONUTRIENTS**: Any deficiency signs to watch + correction
6. **SOIL_TEST_CENTERS**: Where to get soil tested in {d.get('state','N/A')} (government Soil Health Card scheme)
7. **COST_ESTIMATE**: Total fertilizer cost estimate for {d.get('land_area','N/A')} acres

Use Indian fertilizer brand names and current 2024-25 prices where possible.
"""
    },
    {
        "num"   : 4,
        "emoji" : "💧",
        "title" : "Water Management",
        "subtitle": "Irrigation schedule and water requirements",
        "color" : "#0277bd",
        "fields": [
            {"name": "irrigation_method", "label": "Irrigation Method",    "type": "select","required": True,
             "options": ["Select", "Flood/Furrow", "Sprinkler", "Drip", "Rain-fed only"]},
            {"name": "water_availability","label": "Water availability",    "type": "select","required": True,
             "options": ["Abundant (canal/river nearby)", "Moderate (borewell)", "Scarce (rain-fed/shared)"]},
            {"name": "avg_rainfall",      "label": "Avg annual rainfall (mm)","type":"number","required":False,"placeholder":"e.g. 800"},
            {"name": "sowing_date",       "label": "Planned Sowing Date",   "type": "date",  "required": True},
            {"name": "land_slope",        "label": "Land Terrain",          "type": "select","required": False,
             "options": ["Flat", "Slight slope", "Hilly / terrace farming"]},
        ],
        "gemini_key": "irrigation_plan",
        "prompt_fn": lambda d: f"""
You are an irrigation engineer and water management expert for Indian agriculture.

FARM PROFILE:
- Location: {d.get('city','N/A')}, {d.get('state','N/A')}
- Crop: {d.get('confirmed_crop', d.get('crop_recommendation_TOP_CROP','N/A'))}
- Land: {d.get('land_area','N/A')} acres, Terrain: {d.get('land_slope','Flat')}
- Sowing planned: {d.get('sowing_date','N/A')} ({d.get('season','N/A')})
- Irrigation method: {d.get('irrigation_method','N/A')}
- Water availability: {d.get('water_availability','N/A')}
- Annual rainfall: {d.get('avg_rainfall','unknown')} mm
- Water source: {d.get('water_source','N/A')}

Provide a complete irrigation plan:

1. **WATER_REQUIREMENT**: Total water needed for {d.get('confirmed_crop','crop')} on {d.get('land_area','N/A')} acres (liters/MM/season)
2. **IRRIGATION_SCHEDULE**: Week-by-week irrigation schedule from sowing to harvest (critical growth stages highlighted)
3. **METHOD_ADVICE**: Is {d.get('irrigation_method','current method')} optimal? If not, what to switch to and why?
4. **WATER_SAVING_TIPS**: 4 specific water-saving techniques for {d.get('state','N/A')}
5. **RAINFALL_INTEGRATION**: How to use monsoon rains with irrigation for {d.get('state','N/A')}
6. **SUBSIDY_INFO**: Government water/drip irrigation subsidies available in {d.get('state','N/A')} (PM Krishi Sinchayee Yojana etc.)
7. **WARNING_SIGNS**: Signs of over-watering and under-watering for {d.get('confirmed_crop','this crop')}

Include approximate electricity/pump costs for {d.get('land_area','N/A')} acres.
"""
    },
    {
        "num"   : 5,
        "emoji" : "🌦️",
        "title" : "Weather Planning",
        "subtitle": "Seasonal weather forecast and risk planning",
        "color" : "#00695c",
        "fields": [
            {"name": "harvest_month",   "label": "Expected Harvest Month",  "type": "select","required": True,
             "options": ["January","February","March","April","May","June","July","August","September","October","November","December"]},
            {"name": "climate_concern", "label": "Main weather concern",     "type": "select","required": False,
             "options": ["No concern", "Drought/less rain", "Flood/heavy rain", "Frost/cold wave", "Extreme heat", "Cyclone risk"]},
            {"name": "nearest_town",    "label": "Nearest major town/city",  "type": "text",  "required": False, "placeholder": "For weather reference"},
        ],
        "gemini_key": "weather_plan",
        "prompt_fn": lambda d: f"""
You are a meteorologist and agricultural climate expert for India.

FARM DATA:
- Location: {d.get('city','N/A')}, {d.get('state','N/A')} (near {d.get('nearest_town','N/A')})
- Crop: {d.get('confirmed_crop', d.get('crop_recommendation_TOP_CROP','N/A'))}
- Sowing: {d.get('sowing_date','N/A')} | Expected harvest: {d.get('harvest_month','N/A')}
- Season: {d.get('season','N/A')}
- Main weather concern: {d.get('climate_concern','None')}
- Today: {datetime.now().strftime('%B %d, %Y')}

Provide real-time seasonal weather planning advice for a farmer in {d.get('state','India')}:

1. **CURRENT_SEASON_OUTLOOK**: Weather pattern expected for {d.get('state','India')} in {datetime.now().strftime('%B %Y')} onwards (IMD seasonal forecast if known)
2. **CRITICAL_WEATHER_WINDOWS**: 3-4 specific weather windows critical for {d.get('confirmed_crop','this crop')} from sowing to harvest
3. **RISK_ASSESSMENT**: Key weather risks during {d.get('season','this season')} in {d.get('state','India')} and mitigation
4. **FROST_HEAT_ALERTS**: Temperature extremes to watch (with dates/months for {d.get('state','N/A')})
5. **RAIN_WATER_USE**: How to plan around {d.get('state','India')}'s typical rainfall pattern for {d.get('season','this season')}
6. **WEATHER_APPS**: 3 best weather apps/services for Indian farmers (Hindi/regional language options)
7. **CROP_INSURANCE**: Pradhan Mantri Fasal Bima Yojana details for {d.get('state','N/A')} – how to apply, coverage for {d.get('confirmed_crop','this crop')}

Be specific to {d.get('state','India')} climate zone and current conditions as of {datetime.now().strftime('%B %Y')}.
"""
    },
    {
        "num"   : 6,
        "emoji" : "🐛",
        "title" : "Pest & Disease Control",
        "subtitle": "Early prevention and treatment strategy",
        "color" : "#e65100",
        "fields": [
            {"name": "pest_history",  "label": "Past pest/disease problems",  "type": "text",  "required": False, "placeholder": "e.g. Aphids, powdery mildew, or None"},
            {"name": "neighbors_crop","label": "Neighboring field crops",      "type": "text",  "required": False, "placeholder": "e.g. Cotton, Wheat"},
            {"name": "pesticide_pref","label": "Pesticide preference",         "type": "select","required": False,
             "options": ["No preference", "Chemical (fast acting)", "Organic / bio-pesticide", "Integrated Pest Management (IPM)", "Minimum chemicals"]},
            {"name": "spray_equipment","label": "Spray equipment available",   "type": "select","required": False,
             "options": ["Manual knapsack sprayer", "Electric sprayer", "Tractor-mounted sprayer", "Drone spraying (hired)", "None – need advice"]},
        ],
        "gemini_key": "pest_plan",
        "prompt_fn": lambda d: f"""
You are an expert entomologist and plant pathologist for Indian agriculture.

FARM PROFILE:
- Location: {d.get('city','N/A')}, {d.get('state','N/A')}
- Crop: {d.get('confirmed_crop', d.get('crop_recommendation_TOP_CROP','N/A'))}
- Season: {d.get('season','N/A')}, Sowing: {d.get('sowing_date','N/A')}
- Past pest/disease: {d.get('pest_history','None')}
- Neighboring crops: {d.get('neighbors_crop','Unknown')}
- Pesticide preference: {d.get('pesticide_pref','No preference')}
- Spray equipment: {d.get('spray_equipment','Unknown')}
- Current date: {datetime.now().strftime('%B %Y')}

Provide a comprehensive Integrated Pest Management plan:

1. **TOP_THREATS**: 4 major pests/diseases threatening {d.get('confirmed_crop','this crop')} in {d.get('state','India')} during {d.get('season','this season')} – with early symptoms
2. **PREVENTION_CALENDAR**: Month-by-month preventive spray schedule from sowing to harvest
3. **ORGANIC_SOLUTIONS**: Bio-pesticides and natural remedies available in India (neem oil, Beauveria etc.)
4. **CHEMICAL_OPTIONS**: Chemical pesticides (if needed) with names, doses, safety intervals – available in {d.get('state','N/A')}
5. **NEIGHBOR_RISK**: Risk from neighboring {d.get('neighbors_crop','crops')} and how to create buffer
6. **EARLY_WARNING**: 5 visible signs that immediate action is needed
7. **COST_BUDGET**: Estimated pest management cost for {d.get('land_area','N/A')} acres for the full season

Include brand names available in Indian agri shops.
"""
    },
    {
        "num"   : 7,
        "emoji" : "💰",
        "title" : "Market Awareness",
        "subtitle": "Best prices, markets and selling strategy",
        "color" : "#1565c0",
        "fields": [
            {"name": "nearest_mandi",  "label": "Nearest APMC Mandi",          "type": "text",  "required": False, "placeholder": "e.g. Nashik, Pune"},
            {"name": "sell_preference","label": "How do you prefer to sell?",   "type": "select","required": False,
             "options": ["At local mandi", "Direct to trader/broker", "FPO / cooperative", "Online (eNAM)", "Government procurement (MSP)"]},
            {"name": "expected_harvest_qty","label": "Expected yield (quintals)","type":"number","required":False,"placeholder":"e.g. 40", "min": 1},
            {"name": "storage_facility",  "label": "Do you have storage?",      "type": "select","required": False,
             "options": ["No storage", "Home storage (kutcha)", "Pucca warehouse", "Cold storage access", "FPO/cooperative storage"]},
        ],
        "gemini_key": "market_plan",
        "prompt_fn": lambda d: f"""
You are an expert Indian agricultural market analyst.

FARM PROFILE:
- Crop: {d.get('confirmed_crop', d.get('crop_recommendation_TOP_CROP','N/A'))}
- Location: {d.get('city','N/A')}, {d.get('state','N/A')}
- Expected harvest: {d.get('harvest_month','N/A')} | Quantity: {d.get('expected_harvest_qty','unknown')} quintals
- Nearest mandi: {d.get('nearest_mandi','N/A')}
- Selling preference: {d.get('sell_preference','N/A')}
- Storage: {d.get('storage_facility','No storage')}
- Today: {datetime.now().strftime('%B %d, %Y')}

Provide real-time market intelligence:

1. **CURRENT_PRICE**: Current market price of {d.get('confirmed_crop','crop')} in {d.get('state','India')} (₹/quintal, latest available)
2. **MSP_2024_25**: Government MSP for {d.get('confirmed_crop','crop')} for 2024-25 season
3. **PRICE_FORECAST**: Expected price during {d.get('harvest_month','harvest month')} – will it go up or down? Why?
4. **BEST_TIME_TO_SELL**: Optimal selling window to maximize profit
5. **MANDI_OPTIONS**: Top 3 mandis near {d.get('city','N/A')}, {d.get('state','N/A')} with typical prices
6. **DIGITAL_SELLING**: How to use eNAM online portal for {d.get('confirmed_crop','crop')} in {d.get('state','N/A')}
7. **STORAGE_STRATEGY**: If storing, how long to store and expected price appreciation
8. **TOTAL_REVENUE**: Estimated revenue for {d.get('expected_harvest_qty','estimated')} quintals at current prices

Use real current market data. Mention NAFED, FCI, and state agriculture board rates if applicable.
"""
    },
    {
        "num"   : 8,
        "emoji" : "🚜",
        "title" : "Farming Schedule",
        "subtitle": "Step-by-step crop growth management",
        "color" : "#4a148c",
        "fields": [
            {"name": "labor_available",  "label": "Labor available",           "type": "select","required": False,
             "options": ["Family only", "1-2 hired workers", "5+ hired workers", "Mechanized (tractor etc.)", "Mixed"]},
            {"name": "farm_machinery",   "label": "Farm equipment owned",       "type": "text",  "required": False, "placeholder": "e.g. Tractor, pump, or None"},
            {"name": "special_concerns", "label": "Any special concerns",       "type": "text",  "required": False, "placeholder": "e.g. Late monsoon, power cuts"},
        ],
        "gemini_key": "farming_schedule",
        "prompt_fn": lambda d: f"""
You are an expert agricultural extension officer for {d.get('state','India')}.

COMPLETE FARM PROFILE:
- Farmer: {d.get('farmer_name','N/A')}, {d.get('city','N/A')}, {d.get('state','N/A')}
- Crop: {d.get('confirmed_crop', d.get('crop_recommendation_TOP_CROP','N/A'))}
- Land: {d.get('land_area','N/A')} acres, Soil: {d.get('soil_type','N/A')}
- Sowing: {d.get('sowing_date','N/A')}, Harvest: {d.get('harvest_month','N/A')}
- Season: {d.get('season','N/A')}
- Labor: {d.get('labor_available','N/A')}
- Equipment: {d.get('farm_machinery','None')}
- Concerns: {d.get('special_concerns','None')}
- Seed variety: based on Stage 2 advice
- Fertilizer plan: from Stage 3

Create a complete week-by-week farm management calendar:

1. **ACTIVITY_CALENDAR**: Week-by-week activities table (Week 1 to harvest) with columns: Activity | When | What to do | Who/Equipment needed
2. **CRITICAL_MILESTONES**: 5 most important dates/stages not to miss
3. **LABOR_PLAN**: Labor requirement estimate by month (number of workers per day)
4. **MECHANIZATION_TIPS**: Which tasks can be mechanized to save cost for {d.get('land_area','N/A')} acres
5. **GOVT_EXTENSION**: Local KVK (Krishi Vigyan Kendra) or agriculture department contact for {d.get('state','N/A')}
6. **RECORD_KEEPING**: Simple record-keeping template for the farmer to track expenses and activities

Format the activity calendar as a clear table.
"""
    },
    {
        "num"   : 9,
        "emoji" : "📦",
        "title" : "Post-Harvest Management",
        "subtitle": "Storage, transport and reducing losses",
        "color" : "#37474f",
        "fields": [
            {"name": "transport_mode",   "label": "Transport to market",       "type": "select","required": False,
             "options": ["Own vehicle", "Hired truck/tempo", "FPO/cooperative transport", "Rail (for bulk)", "Depends on buyer"]},
            {"name": "packaging_type",   "label": "Packaging used",            "type": "select","required": False,
             "options": ["Jute bags", "PP woven bags", "Loose/bulk", "Crates (for perishables)", "Cardboard (for premium market)"]},
            {"name": "processing_interest","label": "Interested in value addition?","type":"select","required":False,
             "options": ["No – sell raw", "Yes – basic cleaning/grading", "Yes – processing (flour, oil etc.)", "Yes – organic certification"]},
        ],
        "gemini_key": "postharvest_plan",
        "prompt_fn": lambda d: f"""
You are a post-harvest management expert for Indian agriculture.

FARM SUMMARY:
- Crop: {d.get('confirmed_crop', d.get('crop_recommendation_TOP_CROP','N/A'))}
- Location: {d.get('city','N/A')}, {d.get('state','N/A')}
- Expected yield: {d.get('expected_harvest_qty','N/A')} quintals | Harvest: {d.get('harvest_month','N/A')}
- Storage: {d.get('storage_facility','No storage')}
- Transport: {d.get('transport_mode','N/A')}
- Packaging: {d.get('packaging_type','N/A')}
- Value addition interest: {d.get('processing_interest','No')}

Provide practical post-harvest guidance:

1. **HARVEST_TECHNIQUE**: Correct harvesting method for {d.get('confirmed_crop','crop')} to minimize losses
2. **POST_HARVEST_LOSSES**: Main causes of loss for {d.get('confirmed_crop','crop')} and prevention (target: reduce losses to <5%)
3. **STORAGE_GUIDE**: Storage conditions (temperature, humidity, duration) for {d.get('confirmed_crop','crop')}; nearest warehouse/cold storage in {d.get('state','N/A')}
4. **PACKAGING_ADVICE**: Best packaging for {d.get('confirmed_crop','crop')} for {d.get('sell_preference','local market')} selling
5. **GRADING_STANDARDS**: AGMARK or export quality grading standards for {d.get('confirmed_crop','crop')}
6. **VALUE_ADDITION**: {d.get('processing_interest','Basic')} value addition steps with estimated cost and revenue boost
7. **TRANSPORT_OPTIMIZATION**: How to reduce transport cost from {d.get('city','N/A')} to {d.get('nearest_mandi','nearest market')}
8. **GOVT_SCHEMES**: Warehousing schemes (eNWR, WDRA) and cold chain subsidies available in {d.get('state','N/A')}
"""
    },
    {
        "num"   : 10,
        "emoji" : "💵",
        "title" : "Profit & Cost Planning",
        "subtitle": "Full farm economics and final recommendations",
        "color" : "#c62828",
        "fields": [
            {"name": "actual_seed_cost",   "label": "Actual seed cost spent (₹)","type":"number","required":False,"placeholder":"From Stage 2 estimate"},
            {"name": "actual_fert_cost",   "label": "Fertilizer cost (₹)",        "type":"number","required":False,"placeholder":"From Stage 3 estimate"},
            {"name": "labour_cost_budget", "label": "Labor budget (₹)",            "type":"number","required":False,"placeholder":"e.g. 15000"},
            {"name": "other_costs",        "label": "Other costs (₹) – irrigation, pesticide etc.","type":"number","required":False,"placeholder":"e.g. 10000"},
            {"name": "target_profit",      "label": "Target profit (₹)",           "type":"number","required":False,"placeholder":"e.g. 80000"},
        ],
        "gemini_key": "profit_plan",
        "prompt_fn": lambda d: f"""
You are a farm economist and financial advisor for Indian farmers.

COMPLETE 10-STAGE FARM DATA:
- Farmer: {d.get('farmer_name','N/A')}, {d.get('city','N/A')}, {d.get('state','N/A')}
- Crop: {d.get('confirmed_crop', d.get('crop_recommendation_TOP_CROP','N/A'))}
- Land: {d.get('land_area','N/A')} acres | Season: {d.get('season','N/A')}
- Budget: ₹{d.get('budget','N/A')} | Target profit: ₹{d.get('target_profit','N/A')}
- Seed cost: ₹{d.get('actual_seed_cost','estimated')}
- Fertilizer cost: ₹{d.get('actual_fert_cost','estimated')}
- Labor cost: ₹{d.get('labour_cost_budget','estimated')}
- Other costs: ₹{d.get('other_costs','estimated')}
- Expected yield: {d.get('expected_harvest_qty','N/A')} quintals
- Current market price: from Stage 7 analysis
- Harvest month: {d.get('harvest_month','N/A')}

Create a complete farm profit-loss statement and final recommendations:

1. **COST_BREAKDOWN**: Complete cost table (₹ per acre + total for {d.get('land_area','N/A')} acres):
   - Land preparation, Seed, Fertilizer, Irrigation, Labor, Pesticides, Harvest, Transport, Misc
2. **REVENUE_PROJECTION**: Expected revenue at 3 price scenarios (low/medium/high market price)
3. **PROFIT_ESTIMATE**: Net profit/loss estimate and ROI percentage
4. **BREAKEVEN_PRICE**: Minimum price per quintal needed to break even
5. **CASHFLOW_TIMELINE**: Month-by-month cashflow (when money goes out vs. comes in)
6. **LOAN_ADVICE**: If budget is tight, recommend KCC (Kisan Credit Card) or NABARD loans available in {d.get('state','N/A')}
7. **IMPROVEMENT_TIPS**: 3 specific ways to increase profit next season
8. **FINAL_VERDICT**: Overall recommendation: Is this farming plan viable? Risk level: LOW/MEDIUM/HIGH. Confidence score (/10).

Summarize the ENTIRE 10-stage farm plan in a final executive summary paragraph.
"""
    },
]

# ══════════════════════════════════════════════════════════════
#  HELPER: Call Gemini API
# ══════════════════════════════════════════════════════════════
def run_gemini_stage(stage_config, all_data):
    """Call Gemini with the stage prompt and return response text."""
    # Read key lazily — by now load_dotenv() in app.py has already run
    api_key = os.getenv('GOOGLE_API_KEY', '')
    if not api_key:
        return "⚠️ Gemini API key not found. Add GOOGLE_API_KEY to your .env file and restart Flask."
    try:
        genai.configure(api_key=api_key)
        prompt = stage_config['prompt_fn'](all_data)
        gmodel = genai.GenerativeModel('gemini-3-flash-preview')
        response = gmodel.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error stage {stage_config['num']}: {e}")
        traceback.print_exc()
        return f"⚠️ AI analysis error: {str(e)[:200]}. Your inputs have been saved — please continue."

# ══════════════════════════════════════════════════════════════
#  HELPER: PDF Report Generator
# ══════════════════════════════════════════════════════════════
def generate_pdf_report(farm_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('Title', parent=styles['Title'],
        fontSize=22, textColor=colors.HexColor('#1b5e20'),
        spaceAfter=6, fontName='Helvetica-Bold', alignment=TA_CENTER)
    subtitle_style = ParagraphStyle('Sub', parent=styles['Normal'],
        fontSize=11, textColor=colors.HexColor('#555555'),
        spaceAfter=4, alignment=TA_CENTER)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
        fontSize=14, textColor=colors.HexColor('#1b5e20'),
        spaceBefore=14, spaceAfter=6, fontName='Helvetica-Bold')
    h3_style = ParagraphStyle('H3', parent=styles['Heading3'],
        fontSize=11, textColor=colors.HexColor('#0277bd'),
        spaceBefore=8, spaceAfter=4, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
        fontSize=9.5, leading=14, textColor=colors.HexColor('#333333'),
        spaceAfter=6, alignment=TA_JUSTIFY)
    label_style = ParagraphStyle('Label', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor('#666666'), spaceAfter=2)

    story = []
    GREEN = colors.HexColor('#1b5e20')
    LIGHT_GREEN = colors.HexColor('#e8f5e9')
    BLUE = colors.HexColor('#0277bd')

    # ── Cover Page ──────────────────────────────────────────
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("🌱 InfoCrop Farm Plan", title_style))
    story.append(Paragraph("Complete Pre-Crop Information Report", subtitle_style))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=GREEN))
    story.append(Spacer(1, 0.4*cm))

    # Summary table
    crop = farm_data.get('confirmed_crop', farm_data.get('crop_recommendation_TOP_CROP', 'N/A'))
    summary_data = [
        ["Farmer Name",   farm_data.get('farmer_name','N/A'),         "Date",     datetime.now().strftime('%d %b %Y')],
        ["Location",      f"{farm_data.get('city','N/A')}, {farm_data.get('state','N/A')}", "Season", farm_data.get('season','N/A')],
        ["Recommended Crop", crop,                                     "Land Area", f"{farm_data.get('land_area','N/A')} Acres"],
        ["Budget",        f"₹{farm_data.get('budget','N/A')}",        "Sowing Date", farm_data.get('sowing_date','N/A')],
    ]
    t = Table(summary_data, colWidths=[3.5*cm, 6*cm, 3*cm, 5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,-1), LIGHT_GREEN),
        ('TEXTCOLOR',    (0,0), (-1,-1), colors.HexColor('#333')),
        ('FONTNAME',     (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',     (2,0), (2,-1), 'Helvetica-Bold'),
        ('FONTSIZE',     (0,0), (-1,-1), 9),
        ('GRID',         (0,0), (-1,-1), 0.5, colors.HexColor('#c8e6c9')),
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[LIGHT_GREEN, colors.white]),
        ('PADDING',      (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.6*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#c8e6c9')))

    # ── Stage sections ───────────────────────────────────────
    for stage in STAGES:
        story.append(PageBreak())
        snum = stage['num']
        story.append(Paragraph(f"{stage['emoji']} Stage {snum}: {stage['title']}", h2_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#c8e6c9')))
        story.append(Spacer(1, 0.2*cm))

        # Farmer inputs sub-table
        input_rows = [["Field", "Value"]]
        for field in stage['fields']:
            val = farm_data.get(field['name'], '—')
            if val and val != 'Select':
                input_rows.append([field['label'], str(val)])
        if len(input_rows) > 1:
            story.append(Paragraph("Farmer Inputs:", h3_style))
            it = Table(input_rows, colWidths=[6*cm, 11.5*cm])
            it.setStyle(TableStyle([
                ('BACKGROUND',  (0,0), (-1,0), colors.HexColor('#1b5e20')),
                ('TEXTCOLOR',   (0,0), (-1,0), colors.white),
                ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE',    (0,0), (-1,-1), 9),
                ('GRID',        (0,0), (-1,-1), 0.4, colors.HexColor('#ddd')),
                ('ROWBACKGROUNDS',(1,0),(-1,-1),[colors.white, colors.HexColor('#f9f9f9')]),
                ('PADDING',     (0,0), (-1,-1), 5),
            ]))
            story.append(it)
            story.append(Spacer(1, 0.3*cm))

        # Gemini guidance
        gemini_key = stage['gemini_key']
        gemini_text = farm_data.get(f"gemini_{gemini_key}", "")
        if gemini_text:
            story.append(Paragraph("🤖 AI Guidance (Gemini):", h3_style))
            # Split by lines and format
            for line in gemini_text.split('\n'):
                line = line.strip()
                if not line: continue
                if line.startswith('**') and line.endswith('**'):
                    story.append(Paragraph(line.strip('*'), ParagraphStyle('Bold',parent=styles['Normal'],fontName='Helvetica-Bold',fontSize=9.5,spaceBefore=6)))
                elif line.startswith('#'):
                    story.append(Paragraph(line.lstrip('#').strip(), h3_style))
                else:
                    # Clean markdown bold
                    line = line.replace('**','')
                    story.append(Paragraph(line, body_style))
        else:
            story.append(Paragraph("No AI guidance received for this stage.", label_style))

    # ── Final disclaimer ─────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("📋 Important Notes", h2_style))
    story.append(Paragraph(
        "This farm plan was generated using AI analysis based on your inputs and general agricultural knowledge. "
        "Always cross-verify recommendations with your local Krishi Vigyan Kendra (KVK), state agriculture department, "
        "or certified agronomist before making major decisions. Market prices are indicative and subject to change. "
        "Generated by InfoCrop Farm Planner on " + datetime.now().strftime('%d %B %Y at %I:%M %p') + ".",
        body_style
    ))
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=GREEN))
    story.append(Paragraph("InfoCrop — Empowering Indian Farmers with AI", subtitle_style))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════
@farm_bp.route('/', methods=['GET'])
def index():
    """Start page – show Stage 1 form."""
    clear_farm_data()   # wipe any previous plan
    return render_template(
        'FarmPlanner.html',
        stage=STAGES[0],
        stages=STAGES,
        stage_num=1,
        prev_gemini=None,
        farm_data={},
        total_stages=len(STAGES),
        completed=[],
    )


@farm_bp.route('/stage/<int:stage_num>', methods=['POST'])
def process_stage(stage_num):
    """Process the submitted stage form, call Gemini, show next stage."""
    # Load existing data from server-side file (no 4KB cookie limit)
    farm_data = get_farm_data()
    stage_config = STAGES[stage_num - 1]

    # ── Collect form inputs ──────────────────────────────────
    for field in stage_config['fields']:
        val = request.form.get(field['name'], '').strip()
        if val and val not in ('Select',):
            farm_data[field['name']] = val

    # ── Call Gemini for this stage ───────────────────────────
    gemini_response = run_gemini_stage(stage_config, farm_data)
    gemini_key = f"gemini_{stage_config['gemini_key']}"
    farm_data[gemini_key] = gemini_response

    # ── Persist to server-side JSON file ────────────────────
    save_farm_data(farm_data)

    completed = list(range(1, stage_num + 1))

    # ── If all 10 stages done → show report ─────────────────
    if stage_num >= len(STAGES):
        return render_template(
            'FarmPlanner.html',
            stage=None,
            stages=STAGES,
            stage_num=stage_num + 1,
            prev_gemini=gemini_response,
            prev_stage=stage_config,
            farm_data=farm_data,
            total_stages=len(STAGES),
            completed=completed,
            show_report=True,
        )

    # ── Otherwise show next stage ────────────────────────────
    next_stage = STAGES[stage_num]   # 0-indexed
    return render_template(
        'FarmPlanner.html',
        stage=next_stage,
        stages=STAGES,
        stage_num=stage_num + 1,
        prev_gemini=gemini_response,
        prev_stage=stage_config,
        farm_data=farm_data,
        total_stages=len(STAGES),
        completed=completed,
    )


@farm_bp.route('/report', methods=['GET'])
def report():
    """View the full report."""
    farm_data = get_farm_data()
    if not farm_data:
        return redirect(url_for('farm_planner.index'))
    completed = list(range(1, len(STAGES) + 1))
    return render_template(
        'FarmPlanner.html',
        stage=None,
        stages=STAGES,
        stage_num=len(STAGES) + 1,
        farm_data=farm_data,
        total_stages=len(STAGES),
        completed=completed,
        show_report=True,
        prev_gemini=None,
        prev_stage=None,
    )


@farm_bp.route('/download', methods=['GET'])
def download_pdf():
    """Generate and download the PDF farm plan."""
    farm_data = get_farm_data()
    if not farm_data:
        return redirect(url_for('farm_planner.index'))
    try:
        pdf_buffer = generate_pdf_report(farm_data)
        farmer_name = farm_data.get('farmer_name', 'Farm').replace(' ', '_')
        filename = f"InfoCrop_FarmPlan_{farmer_name}_{datetime.now().strftime('%Y%m%d')}.pdf"
        response = make_response(pdf_buffer.read())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
    except Exception as e:
        traceback.print_exc()
        return f"PDF generation error: {e}", 500


@farm_bp.route('/restart', methods=['GET'])
def restart():
    clear_farm_data()
    return redirect(url_for('farm_planner.index'))
