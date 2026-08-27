"""
Laboratory Activity 1: Data Cleaning Pipeline
Domain: Aquatic Chemistry & Public Health
Project: Water Potability and Chemical Safety Assessment
Dataset: water_potability.csv
"""

import pandas as pd
import numpy as np
import os

def clean_water_potability(input_path, output_path):
    print("=" * 70)
    print("LABORATORY ACTIVITY 1: DATA CLEANING PIPELINE")
    print("Project: Water Potability and Chemical Safety Assessment")
    print("=" * 70)
    
    # 1. Load raw dataset
    df_raw = pd.read_csv(input_path)
    print(f"\n[RAW DATASET PROFILE]")
    print(f"Total Rows: {len(df_raw):,}")
    print(f"Total Columns: {df_raw.shape[1]}")
    print(f"Total Missing Values: {df_raw.isna().sum().sum():,}")
    print(f"Missing Sulfate: {df_raw['Sulfate'].isna().sum()} ({df_raw['Sulfate'].isna().mean()*100:.2f}%)")
    print(f"Missing ph: {df_raw['ph'].isna().sum()} ({df_raw['ph'].isna().mean()*100:.2f}%)")
    print(f"Missing Trihalomethanes: {df_raw['Trihalomethanes'].isna().sum()} ({df_raw['Trihalomethanes'].isna().mean()*100:.2f}%)")
    
    df = df_raw.copy()
    
    # -------------------------------------------------------------
    # Step 1: Enforce Physicochemical pH Scale Boundaries (0 to 14)
    # -------------------------------------------------------------
    invalid_ph_mask = (df["ph"] < 0) | (df["ph"] > 14)
    invalid_ph_count = invalid_ph_mask.sum()
    print(f"\n[Step 1] Sanitizing pH physical scale (0 to 14). Out-of-bounds detected: {invalid_ph_count}")
    df.loc[invalid_ph_mask, "ph"] = np.nan
    
    # -------------------------------------------------------------
    # Step 2: Class-Conditional Median Imputation for Missing Assays
    # -------------------------------------------------------------
    print("\n[Step 2] Applying Class-Conditional Median Imputation for missing chemical assays...")
    impute_features = ["ph", "Sulfate", "Trihalomethanes"]
    for col in impute_features:
        df[col] = df.groupby("Potability")[col].transform(
            lambda group: group.fillna(group.median())
        )
        
    # If any residual nulls remain (e.g. if an entire group was empty), fill with global median
    for col in df.columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # -------------------------------------------------------------
    # Step 3: Extreme Outlier Filtering on Total Dissolved Solids
    # -------------------------------------------------------------
    # Use 3.0*IQR threshold to preserve valid brackish groundwater while trimming sensor spikes
    q1 = df["Solids"].quantile(0.25)
    q3 = df["Solids"].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + 3.0 * iqr
    extreme_solids_count = (df["Solids"] > upper_limit).sum()
    print(f"\n[Step 3] Handling extreme solids (> 3*IQR = {upper_limit:.1f} ppm): Pruning {extreme_solids_count} extreme sensor spikes.")
    df = df[df["Solids"] <= upper_limit].reset_index(drop=True)
    
    # -------------------------------------------------------------
    # Step 4: Data Type Integrity & Precision Normalization
    # -------------------------------------------------------------
    print("\n[Step 4] Enforcing explicit data types and continuous metric precision...")
    df["Potability"] = df["Potability"].astype(int)
    
    continuous_cols = [c for c in df.columns if c != "Potability"]
    for col in continuous_cols:
        df[col] = df[col].round(3)
        
    assert df.isna().sum().sum() == 0, "Dataset still contains unresolved missing values!"
    
    # -------------------------------------------------------------
    # Step 5: Export Cleaned Dataset
    # -------------------------------------------------------------
    df.to_csv(output_path, index=False)
    print(f"\n[EXPORT] Successfully saved cleaned dataset to: {output_path}")
    
    print("\n" + "=" * 70)
    print("BEFORE VS AFTER DATA CLEANING COMPARISON")
    print("=" * 70)
    summary_table = pd.DataFrame({
        "Metric / Check": [
            "Total Rows",
            "Total Missing Values",
            "Missing Sulfate",
            "Missing pH",
            "Missing Trihalomethanes",
            "Max Solids (ppm)",
            "Potable Samples Count (%)"
        ],
        "Raw Dataset": [
            f"{len(df_raw):,}",
            f"{df_raw.isna().sum().sum():,}",
            f"{df_raw['Sulfate'].isna().sum()} ({df_raw['Sulfate'].isna().mean()*100:.2f}%)",
            f"{df_raw['ph'].isna().sum()} ({df_raw['ph'].isna().mean()*100:.2f}%)",
            f"{df_raw['Trihalomethanes'].isna().sum()} ({df_raw['Trihalomethanes'].isna().mean()*100:.2f}%)",
            f"{df_raw['Solids'].max():.1f}",
            f"{df_raw['Potability'].sum()} ({df_raw['Potability'].mean()*100:.2f}%)"
        ],
        "Cleaned Dataset": [
            f"{len(df):,}",
            f"{df.isna().sum().sum():,}",
            f"{df['Sulfate'].isna().sum()} (0.00%)",
            f"{df['ph'].isna().sum()} (0.00%)",
            f"{df['Trihalomethanes'].isna().sum()} (0.00%)",
            f"{df['Solids'].max():.1f}",
            f"{df['Potability'].sum()} ({df['Potability'].mean()*100:.2f}%)"
        ]
    })
    print(summary_table.to_string(index=False))
    print("=" * 70)
    return df

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "water_potability.csv")
    output_file = os.path.join(script_dir, "water_potability_cleaned.csv")
    clean_water_potability(input_file, output_file)
