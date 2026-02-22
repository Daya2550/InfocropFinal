"""
Crop Price Prediction Model Training Script  (v2 — Clean Data)
===============================================================
Uses crop_price_dataset_clean.csv (all corrupted rows removed).
Trains a RandomForestRegressor on ALL 31 dataset features.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CLEAN_CSV = os.path.join(BASE_DIR, "crop_price_dataset_clean.csv")
RAW_CSV   = os.path.join(BASE_DIR, "crop_price_dataset.csv")

print("=" * 65)
print("CROP PRICE MODEL TRAINING  (All Features, Clean Data)")
print("=" * 65)

# ── 1. Load clean dataset ────────────────────────────────────────
csv_path = CLEAN_CSV if os.path.exists(CLEAN_CSV) else RAW_CSV
print(f"\n[1] Loading: {os.path.basename(csv_path)}")
df = pd.read_csv(csv_path)
print(f"    Shape: {df.shape}")

# Hard-cast Previous_30_Day_Avg_Price to numeric just in case
df['Previous_30_Day_Avg_Price'] = pd.to_numeric(
    df['Previous_30_Day_Avg_Price'], errors='coerce'
)

# Keep only valid Price_Trend values
valid_trends = ['Increasing', 'Decreasing', 'Stable']
df['Price_Trend'] = df['Price_Trend'].astype(str).str.strip()
df = df[df['Price_Trend'].isin(valid_trends)]

# Drop remaining nulls
df.dropna(subset=['Price'], inplace=True)
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)
print(f"    Rows after cleaning: {len(df):,}")

# ── 2. Feature Engineering (Date) ───────────────────────────────
print("\n[2] Engineering date features ...")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Day']         = df['Date'].dt.day
df['Month']       = df['Date'].dt.month
df['Year']        = df['Date'].dt.year
df['Week_Number'] = df['Date'].dt.isocalendar().week.astype(int)
df.drop(columns=['Date'], inplace=True)

# ── 3. Define ALL features ───────────────────────────────────────
TARGET = 'Price'
FEATURE_COLS = [c for c in df.columns if c != TARGET]

print(f"\n[3] All features used ({len(FEATURE_COLS)}):")
for i, col in enumerate(FEATURE_COLS, 1):
    print(f"    {i:2d}. {col}")

X = df[FEATURE_COLS].copy()
y = df[TARGET].copy()

# ── 4. Encode categoricals ───────────────────────────────────────
print("\n[4] Encoding categorical columns ...")
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le
    print(f"    {col}: {le.classes_.tolist()[:6]}{'...' if len(le.classes_) > 6 else ''}")

# ── 5. Train/Test Split ──────────────────────────────────────────
print("\n[5] Splitting data (80/20) ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"    Train: {X_train.shape}  Test: {X_test.shape}")

# ── 6. Train Model ───────────────────────────────────────────────
print("\n[6] Training RandomForestRegressor (200 trees) ...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("    Done!")

# ── 7. Evaluate ──────────────────────────────────────────────────
print("\n[7] Evaluating ...")
y_pred = model.predict(X_test)
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"    R² Score : {r2:.4f}  ({r2*100:.1f}%)")
print(f"    MAE      : ₹{mae:.2f} per quintal")
print(f"    RMSE     : ₹{rmse:.2f}")
print(f"    Price range in dataset: ₹{y.min():.0f} – ₹{y.max():.0f}")

# Feature importances
importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print("\n    Top 15 Feature Importances:")
for feat, imp in importances.head(15).items():
    bar = "█" * int(imp * 50)
    print(f"      {feat:35s}: {imp*100:5.2f}%  {bar}")

# ── 8. Save Artifacts ────────────────────────────────────────────
print("\n[8] Saving model artifacts ...")
joblib.dump(model,        os.path.join(BASE_DIR, "model.pkl"))
joblib.dump(encoders,     os.path.join(BASE_DIR, "encoder.pkl"))
joblib.dump(FEATURE_COLS, os.path.join(BASE_DIR, "columns.pkl"))

print(f"    model.pkl   → saved ({len(FEATURE_COLS)} features)")
print(f"    encoder.pkl → saved ({len(encoders)} encoders)")
print(f"    columns.pkl → saved")

print(f"\n{'='*65}")
print(f"✔ TRAINING COMPLETE")
print(f"  R²={r2*100:.1f}%  |  MAE=₹{mae:.2f}  |  RMSE=₹{rmse:.2f}")
print(f"{'='*65}")
