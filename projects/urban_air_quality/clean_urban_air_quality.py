"""
Laboratory Activity 1: Data Cleaning Pipeline
Domain: Atmospheric Science & Environmental Health
Project: Urban Air Quality Assessment
Dataset: city_day.csv
"""

import pandas as pd
import numpy as np
import os

def classify_aqi(aqi):
    if pd.isna(aqi):
        return "Unknown"
    elif aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

def clean_urban_air_quality(input_path, output_path):
    print("=" * 70)
    print("LABORATORY ACTIVITY 1: DATA CLEANING PIPELINE")
    print("Project: Urban Air Quality Assessment")
    print("=" * 70)
    
    # 1. Load raw dataset
    df_raw = pd.read_csv(input_path)
    print(f"\n[RAW DATASET PROFILE]")
    print(f"Total Rows: {len(df_raw):,}")
    print(f"Total Columns: {df_raw.shape[1]}")
    print(f"Total Missing Values: {df_raw.isna().sum().sum():,}")
    print(f"Missing PM2.5: {df_raw['PM2.5'].isna().sum():,} ({df_raw['PM2.5'].isna().mean()*100:.2f}%)")
    print(f"Missing AQI: {df_raw['AQI'].isna().sum():,} ({df_raw['AQI'].isna().mean()*100:.2f}%)")
    
    df = df_raw.copy()
    
    # -------------------------------------------------------------
    # Step 1: Datatype Conversion and Temporal Sorting
    # -------------------------------------------------------------
    print("\n[Step 1] Converting Date to datetime and sorting chronologically by City & Date...")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df = df.sort_values(by=["City", "Date"]).reset_index(drop=True)
    
    # -------------------------------------------------------------
    # Step 2: Categorical and Text Normalization
    # -------------------------------------------------------------
    print("\n[Step 2] Standardizing city names and categorical text formatting...")
    df["City"] = df["City"].str.strip().str.title()
    if "AQI_Bucket" in df.columns:
        df["AQI_Bucket"] = df["AQI_Bucket"].astype(str).str.strip().str.title()
        df.loc[df["AQI_Bucket"] == "Nan", "AQI_Bucket"] = np.nan
        
    # -------------------------------------------------------------
    # Step 3: Handling Physically Impossible Negative Concentrations
    # -------------------------------------------------------------
    pollutant_columns = [
        "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", 
        "Benzene", "Toluene", "Xylene"
    ]
    
    negative_counts = 0
    for col in pollutant_columns:
        neg_mask = df[col] < 0
        neg_in_col = neg_mask.sum()
        if neg_in_col > 0:
            negative_counts += neg_in_col
            df.loc[neg_mask, col] = np.nan
            
    print(f"\n[Step 3] Sanitizing sensor drift: Replaced {negative_counts} negative concentration values with NaN.")
    
    # -------------------------------------------------------------
    # Step 4: Domain-Aware Time-Series Imputation
    # -------------------------------------------------------------
    print("\n[Step 4] Applying City-Grouped Time-Series Linear Interpolation (limit=3) + Bounded Fill...")
    
    # City-level time-continuity interpolation
    df[pollutant_columns] = df.groupby("City")[pollutant_columns].transform(
        lambda group: group.interpolate(method="linear", limit=3).ffill().bfill()
    )
    
    # Impute AQI using city-grouped temporal interpolation
    df["AQI"] = df.groupby("City")["AQI"].transform(
        lambda group: group.interpolate(method="linear", limit=3).ffill().bfill()
    )
    
    # Secondary national median fallback for smaller stations lacking specific gas analyzers (e.g. Xylene)
    for col in pollutant_columns + ["AQI"]:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # -------------------------------------------------------------
    # Step 5: Recalculate & Harmonize AQI Classification Buckets
    # -------------------------------------------------------------
    print("\n[Step 5] Recalculating and standardizing AQI_Bucket according to national CPCB standards...")
    df["AQI_Bucket"] = df["AQI"].apply(classify_aqi)
    
    # Round numerical metrics to 2 decimals
    for col in pollutant_columns + ["AQI"]:
        df[col] = df[col].round(2)
        
    assert df.isna().sum().sum() == 0, "Dataset contains unresolved missing values!"
    
    # -------------------------------------------------------------
    # Step 6: Export Cleaned Dataset
    # -------------------------------------------------------------
    df.to_csv(output_path, index=False)
    print(f"\n[EXPORT] Successfully saved cleaned dataset to: {output_path}")
    
    print("\n" + "=" * 70)
    print("BEFORE VS AFTER DATA CLEANING COMPARISON")
    print("=" * 70)
    summary_table = pd.DataFrame({
        "Metric / Check": [
            "Total Rows",
            "Date Format",
            "Total Missing Values",
            "Missing PM2.5",
            "Missing AQI",
            "Missing AQI_Bucket",
            "Negative Sensor Values",
            "Maximum PM2.5 (Retained Fog Outliers)"
        ],
        "Raw Dataset": [
            f"{len(df_raw):,}",
            "string (object)",
            f"{df_raw.isna().sum().sum():,}",
            f"{df_raw['PM2.5'].isna().sum():,} ({df_raw['PM2.5'].isna().mean()*100:.2f}%)",
            f"{df_raw['AQI'].isna().sum():,} ({df_raw['AQI'].isna().mean()*100:.2f}%)",
            f"{df_raw['AQI_Bucket'].isna().sum():,} ({df_raw['AQI_Bucket'].isna().mean()*100:.2f}%)",
            f"{negative_counts}",
            f"{df_raw['PM2.5'].max():.1f} ug/m3"
        ],
        "Cleaned Dataset": [
            f"{len(df):,}",
            "datetime64 (YYYY-MM-DD)",
            f"{df.isna().sum().sum():,}",
            f"{df['PM2.5'].isna().sum()} (0.00%)",
            f"{df['AQI'].isna().sum()} (0.00%)",
            f"{df['AQI_Bucket'].isna().sum()} (0.00%)",
            "0",
            f"{df['PM2.5'].max():.1f} ug/m3"
        ]
    })
    print(summary_table.to_string(index=False))
    print("=" * 70)
    return df

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "city_day.csv")
    output_file = os.path.join(script_dir, "city_day_cleaned.csv")
    clean_urban_air_quality(input_file, output_file)
