import joblib
import pandas as pd

try:
    encoders = joblib.load("encoder.pkl")
    print("Keys in encoder.pkl:", encoders.keys())
    
    if "Price_Trend" in encoders:
        print("Price_Trend classes:", encoders["Price_Trend"].classes_)
    else:
        print("Price_Trend NOT found in encoder.pkl")
        
    model_columns = joblib.load("columns.pkl")
    print("Model columns:", model_columns)
    
except Exception as e:
    print(f"Error loading artifacts: {e}")
