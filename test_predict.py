import pandas as pd
import joblib
import os
import sys

def test_prediction():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Base Dir: {base_dir}")
        
        model_path = os.path.join(base_dir, "model.pkl")
        encoder_path = os.path.join(base_dir, "encoder.pkl")
        columns_path = os.path.join(base_dir, "columns.pkl")
        
        print(f"Loading from {model_path}...")
        model = joblib.load(model_path)
        encoders = joblib.load(encoder_path)
        model_columns = joblib.load(columns_path)
        print("Model loaded successfully.")
        
        # Create dummy input based on valid values from csv (e.g. line 2)
        # 2022-09-29,Tomato,Madhya Pradesh,Indore,Depalpur,Kharif,3206.53,920.3,167.8,31.0,26.5,83.4,0
        input_data = {
            "Date": "2022-09-29",
            "Crop_Name": "Tomato",
            "State": "Madhya Pradesh",
            "District": "Indore",
            "Market_Name": "Depalpur",
            "Season": "Kharif",
            "Previous_Day_Price": 3206.53,
            "Market_Arrival_Quantity": 920.3,
            "Rainfall": 167.8,
            "Temperature_Max": 31.0,
            "Temperature_Min": 26.5,
            "Humidity": 83.4,
            "Festival_Flag": 0,
            "Previous_7_Day_Avg_Price": 3100.0,
            "Previous_30_Day_Avg_Price": 3050.0,
            "Price_Trend": "Increasing",
            "Production_Quantity": 15000.0,
            "Yield_Per_Hectare": 4.5,
            "Stock_Available": 500.0,
            "Soil_Moisture": 45.0,
            "Demand_Index": 110.0,
            "Min_Price": 3000.0,
            "Max_Price": 3500.0,
            "Modal_Price": 3200.0,
            "Number_of_Buyers": 45,
            "Fuel_Price": 98.0,
            "Transportation_Cost": 450.0,
            "Inflation_Rate": 5.5,
            "Harvest_Season_Flag": 1
        }
        
        # Calculate Week_Number dynamically
        dt = pd.to_datetime(input_data["Date"])
        input_data["Week_Number"] = dt.week
        
        df = pd.DataFrame([input_data])
        
        print("Processing input...")
        df["Date"] = pd.to_datetime(df["Date"])
        df["Day"] = df["Date"].dt.day
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year
        df = df.drop(columns=["Date"])
        
        print("Encoding...")
        for col in encoders:
            if col in df.columns:
                print(f"Encoding {col}...")
                df[col] = encoders[col].transform(df[col].astype(str))
                
        df = df[model_columns]
        
        print("Predicting...")
        prediction = model.predict(df)[0]
        print(f"Prediction result: {prediction}")
        
    except Exception as e:
        print(f"CAUGHT EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()
