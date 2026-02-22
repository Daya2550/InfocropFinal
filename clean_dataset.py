"""
Dataset Cleaning Script for Crop Price Prediction
===================================================
Removes corrupted rows, fixes swapped columns, validates price ranges,
and produces a clean CSV ready for model training.
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_CSV   = os.path.join(BASE_DIR, "crop_price_dataset.csv")
CLEAN_CSV = os.path.join(BASE_DIR, "crop_price_dataset_clean.csv")

# ── Realistic price bounds per crop (₹/quintal) ─────────────────────────────
PRICE_BOUNDS = {
    'Tomato': (50,   6000),
    'Wheat':  (1200, 5000),
    'Potato': (50,   3500),
    'Onion':  (150,  8000),
    'Rice':   (1200, 10000),
}
DEFAULT_BOUNDS = (50, 12000)   # fallback for any other crop

VALID_PRICE_TRENDS = {'Increasing', 'Decreasing', 'Stable'}
VALID_SEASONS      = {'Rabi', 'Kharif', 'Zaid'}

print("=" * 65)
print("DATASET CLEANING PIPELINE")
print("=" * 65)

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_CSV)
print(f"\nLoaded  : {len(df):,} rows × {len(df.columns)} columns")

# ── Fix: First 3 rows sometimes have a header-echo — drop if any ─────────────
df = df[df['Price'].astype(str).str.strip() != 'Price']

# ── Fix: Previous_30_Day_Avg_Price — must be numeric ─────────────────────────
before = len(df)
df['Previous_30_Day_Avg_Price'] = pd.to_numeric(df['Previous_30_Day_Avg_Price'], errors='coerce')
bad_30 = df['Previous_30_Day_Avg_Price'].isnull().sum()
print(f"\nBad Previous_30_Day_Avg_Price (non-numeric) : {bad_30} rows → dropped")
df.dropna(subset=['Previous_30_Day_Avg_Price'], inplace=True)
print(f"Rows after fixing Prev-30 : {len(df):,}")

# ── Fix: Price_Trend — must be valid category ─────────────────────────────────
bad_trend = ~df['Price_Trend'].astype(str).str.strip().isin(VALID_PRICE_TRENDS)
print(f"\nBad Price_Trend (not Increasing/Decreasing/Stable) : {bad_trend.sum()} rows → dropped")
df = df[~bad_trend]
df['Price_Trend'] = df['Price_Trend'].str.strip()
print(f"Rows after fixing Price_Trend : {len(df):,}")

# ── Fix: Season — must be valid ───────────────────────────────────────────────
bad_season = ~df['Season'].astype(str).str.strip().isin(VALID_SEASONS)
print(f"\nBad Season : {bad_season.sum()} rows → dropped")
df = df[~bad_season]

# ── Fix: Price — drop nulls, drop impossible values ──────────────────────────
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
before_price = len(df)
df.dropna(subset=['Price'], inplace=True)

# Apply per-crop realistic bounds
def check_price(row):
    lo, hi = PRICE_BOUNDS.get(row['Crop_Name'], DEFAULT_BOUNDS)
    return lo <= row['Price'] <= hi

bad_price_mask = ~df.apply(check_price, axis=1)
print(f"\nPrice outside realistic bounds : {bad_price_mask.sum()} rows → dropped")
df = df[~bad_price_mask]

# ── Fix: Key numeric columns — coerce & drop if still null ───────────────────
numeric_cols = [
    'Previous_Day_Price', 'Market_Arrival_Quantity', 'Rainfall',
    'Temperature_Max', 'Temperature_Min', 'Humidity',
    'Previous_7_Day_Avg_Price', 'Production_Quantity',
    'Yield_Per_Hectare', 'Stock_Available', 'Soil_Moisture',
    'Demand_Index', 'Number_of_Buyers', 'Min_Price', 'Max_Price',
    'Modal_Price', 'Fuel_Price', 'Transportation_Cost', 'Inflation_Rate',
    'Week_Number'
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        nulls = df[col].isnull().sum()
        if nulls > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Imputed {nulls} nulls in '{col}' with median {median_val:.2f}")

# ── Fix: Date — parse, drop unparseable ─────────────────────────────────────
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
bad_dates = df['Date'].isnull().sum()
if bad_dates:
    print(f"\nBad dates : {bad_dates} rows → dropped")
    df.dropna(subset=['Date'], inplace=True)

# Reconstruct consistent date-string format
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

# ── Fix: Boolean flags ────────────────────────────────────────────────────────
for flag_col in ['Festival_Flag', 'Harvest_Season_Flag']:
    if flag_col in df.columns:
        df[flag_col] = pd.to_numeric(df[flag_col], errors='coerce').fillna(0).astype(int).clip(0, 1)

# ── Fix: Previous_Day_Price sanity — must be > 0 ─────────────────────────────
bad_prev = df['Previous_Day_Price'] <= 0
if bad_prev.sum():
    print(f"\nNon-positive Previous_Day_Price : {bad_prev.sum()} rows → replaced with overall median")
    med = df.loc[~bad_prev, 'Previous_Day_Price'].median()
    df.loc[bad_prev, 'Previous_Day_Price'] = med

# ── Fix: Validate Previous prices vs. Price (sanity cross-check) ────────────
df['_ratio_7']  = df['Previous_7_Day_Avg_Price']  / df['Previous_Day_Price'].clip(lower=1)
df['_ratio_30'] = df['Previous_30_Day_Avg_Price'] / df['Previous_Day_Price'].clip(lower=1)

# Extremely abnormal ratios (> 20x or < 0.05x) indicate swapped/garbage values
bad_ratios = (df['_ratio_7'] > 20) | (df['_ratio_7'] < 0.05) | \
             (df['_ratio_30'] > 20) | (df['_ratio_30'] < 0.05)
print(f"\nAbnormal price ratios (7d/30d vs prev-day) : {bad_ratios.sum()} rows → dropped")
df = df[~bad_ratios]
df.drop(columns=['_ratio_7', '_ratio_30'], inplace=True)

# ── Drop empty categorical fields ────────────────────────────────────────────
for cat_col in ['Crop_Name', 'State', 'District', 'Market_Name', 'Season']:
    df = df[df[cat_col].astype(str).str.strip() != '']

# ── Final report ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"CLEAN DATASET SUMMARY")
print(f"{'='*65}")
print(f"  Rows retained : {len(df):,} / {before}")
print(f"  Rows removed  : {before - len(df):,}")
print(f"  Columns       : {len(df.columns)}")
print(f"\n  Crops  : {df['Crop_Name'].unique().tolist()}")
print(f"  States : {df['State'].unique().tolist()}")
print(f"  Seasons: {df['Season'].unique().tolist()}")
print(f"  Price_Trend: {df['Price_Trend'].unique().tolist()}")
print(f"\n  Price stats by crop:")
print(df.groupby('Crop_Name')['Price'].agg(['min','max','mean','count']).round(2).to_string())
print(f"\n  Remaining nulls: {df.isnull().sum().sum()}")

# ── Save ─────────────────────────────────────────────────────────────────────
df.to_csv(CLEAN_CSV, index=False)
print(f"\n✔  Clean dataset saved → {CLEAN_CSV}")
print("=" * 65)
