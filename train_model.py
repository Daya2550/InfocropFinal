import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("crop_price_dataset.csv")

# -------- Feature Selection & Engineering --------

# Select specific columns requests by user + Target (Price)
# Date, Crop, State, District, Market, Season, Previous Price, Market Arrival, 
# Rainfall, Temperature Max, Temperature Min, Humidity, Festival
# + NEW: Previous_7_Day_Avg_Price, Previous_30_Day_Avg_Price, Price_Trend
# + Production_Quantity, Yield_Per_Hectare, Stock_Available
# + Soil_Moisture, Demand_Index, Min_Price, Max_Price, Modal_Price, Number_of_Buyers
# + Fuel_Price, Transportation_Cost, Inflation_Rate, Harvest_Season_Flag, Week_Number
required_columns = [
    "Date", "Crop_Name", "State", "District", "Market_Name", "Season", 
    "Previous_Day_Price", "Market_Arrival_Quantity", "Rainfall", 
    "Temperature_Max", "Temperature_Min", "Humidity", "Festival_Flag",
    "Previous_7_Day_Avg_Price", "Previous_30_Day_Avg_Price", "Price_Trend",
    "Production_Quantity", "Yield_Per_Hectare", "Stock_Available",
    "Soil_Moisture", "Demand_Index", "Min_Price", "Max_Price", "Modal_Price",
    "Number_of_Buyers", "Fuel_Price", "Transportation_Cost", "Inflation_Rate",
    "Harvest_Season_Flag", "Week_Number",
    "Price"
]

# Ensure we only use available columns (intersect with csv columns to avoid errors if csv is missing some)
# But for this task, we assume CSV has them or we made dummy data with them.
# The dummy data I created previously has all these.

data = data[required_columns]

# Convert date to datetime
data["Date"] = pd.to_datetime(data["Date"])

# Extract date features for training
data["Day"] = data["Date"].dt.day
data["Month"] = data["Date"].dt.month
data["Year"] = data["Date"].dt.year

# Encode categorical columns
categorical_cols = [
    "Crop_Name",
    "State",
    "District",
    "Market_Name",
    "Season",
    "Price_Trend"
]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    # Ensure string type for encoding
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le

# Features and target
# Drop Price (target) and Date (original datetime object)
X = data.drop(columns=["Price", "Date"])
y = data["Price"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Save model, encoders, and columns
joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoder.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("Model trained successfully with refined features")
