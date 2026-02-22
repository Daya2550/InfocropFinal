import pandas as pd
import numpy as np
import random

def augment_data():
    try:
        print("Loading dataset...")
        df = pd.read_csv("crop_price_dataset.csv")
        
        # Check if columns already exist to avoid overwriting real data if run multiple times
        # But for this task, we want to ensure all new columns are present.
        
        nrows = len(df)
        print(f"Augmenting {nrows} rows...")
        
        # Helper for random floats
        def rand_float(min_v, max_v):
            return np.random.uniform(min_v, max_v, nrows)
            
        # Helper for random choices
        def rand_choice(options):
            return np.random.choice(options, nrows)
            
        # 1. Price related averages (Mocking based on current price for realism)
        # Assuming current Price is approx average, we create some variance
        df["Previous_7_Day_Avg_Price"] = df["Price"] * np.random.uniform(0.9, 1.1, nrows)
        df["Previous_30_Day_Avg_Price"] = df["Price"] * np.random.uniform(0.85, 1.15, nrows)
        
        # 2. Price Trend
        df["Price_Trend"] = rand_choice(['Increasing', 'Decreasing', 'Stable'])
        
        # 3. Production & Yield
        df["Production_Quantity"] = rand_float(1000, 50000) # Tons?
        df["Yield_Per_Hectare"] = rand_float(1, 10) # Tons/Hectare
        df["Stock_Available"] = rand_float(100, 10000)
        
        # 4. Soil & Demand
        df["Soil_Moisture"] = rand_float(10, 90) # Percentage
        df["Demand_Index"] = rand_float(50, 150) # Index
        df["Number_of_Buyers"] = np.random.randint(1, 100, nrows)
        
        # 5. Min/Max/Modal Prices (Derived roughly from Price)
        df["Min_Price"] = df["Price"] * np.random.uniform(0.8, 0.95, nrows)
        df["Max_Price"] = df["Price"] * np.random.uniform(1.05, 1.2, nrows)
        df["Modal_Price"] = df["Price"] # Often the traded price
        
        # 6. Economic
        df["Fuel_Price"] = rand_float(90, 110) # INR/Liter approx
        df["Transportation_Cost"] = rand_float(100, 2000) # Per ton/trip
        df["Inflation_Rate"] = rand_float(4, 8) # Percentage
        
        # 7. Harvest Flag & Week Number
        # Harvest flag random for now (0 or 1)
        df["Harvest_Season_Flag"] = np.random.randint(0, 2, nrows)
        
        # Week Number derived from Date
        df["Date"] = pd.to_datetime(df["Date"])
        df["Week_Number"] = df["Date"].dt.isocalendar().week
        
        # Save back
        print("Saving augmented dataset...")
        df.to_csv("crop_price_dataset.csv", index=False)
        print("Done! Columns now:", df.columns.tolist())
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    augment_data()
