"""
clean_water_potability.py
-------------------------
Standalone Python script for Laboratory Activity 1: Data Cleaning Pipeline
Domain: Water Potability and Chemical Safety Assessment
Course: DS311 - Exploratory Data Analysis
"""

import pandas as pd
import numpy as np

def clean_water_dataset(input_csv="water_potability.csv", output_csv="water_potability_cleaned.csv"):
    print(f"Loading raw dataset from '{input_csv}'...")
    df = pd.read_csv(input_csv)
    print(f"Initial dimensions: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # 1. Audit and drop duplicate records
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Removed {dup_count} duplicate records.")
    else:
        print("Deduplication check: 0 duplicates found.")

    # 2. Enforce pH boundaries (0 <= pH <= 14)
    invalid_ph = (df["ph"] < 0) | (df["ph"] > 14)
    invalid_ph_count = invalid_ph.sum()
    if invalid_ph_count > 0:
        df.loc[invalid_ph, "ph"] = np.nan
        print(f"Sanitized {invalid_ph_count} impossible pH readings to NaN.")
    else:
        print("pH boundary check: All readings within valid 0 to 14 range.")

    # 3. Class-Conditional Median Imputation (by Potability group)
    impute_features = ["ph", "Sulfate", "Trihalomethanes"]
    for col in impute_features:
        median_potable = df[df['Potability'] == 1][col].median()
        median_non_potable = df[df['Potability'] == 0][col].median()
        print(f"[{col}] Imputing medians -> Potable (1): {median_potable:.3f} | Non-Potable (0): {median_non_potable:.3f}")
        df[col] = df.groupby("Potability")[col].transform(lambda g: g.fillna(g.median()))

    # Safety check for any remaining missing values
    for col in df.columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # 4. Outlier check on Total Dissolved Solids (3.0 * IQR extreme cutoff)
    q1 = df["Solids"].quantile(0.25)
    q3 = df["Solids"].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + 3.0 * iqr
    
    outliers_pruned = (df["Solids"] > upper_limit).sum()
    df = df[df["Solids"] <= upper_limit].reset_index(drop=True)
    print(f"Solids extreme cutoff (3.0 * IQR): {upper_limit:.2f} ppm. Pruned: {outliers_pruned} records.")

    # 5. Data formatting and precision
    df["Potability"] = df["Potability"].astype(int)
    continuous_cols = [c for c in df.columns if c != "Potability"]
    for col in continuous_cols:
        df[col] = df[col].round(3)

    # 6. Integrity checks
    assert df.isna().sum().sum() == 0, "Assertion Error: Missing values remain!"
    assert df["Potability"].isin([0, 1]).all(), "Assertion Error: Potability contains non-binary values!"
    assert df.duplicated().sum() == 0, "Assertion Error: Duplicates detected!"

    # 7. Export Cleaned Dataset
    df.to_csv(output_csv, index=False)
    print(f"Data cleaning pipeline completed. Exported to '{output_csv}'.")
    print(f"Final shape: {df.shape[0]:,} rows, {df.shape[1]} columns. Total missing: 0.")
    return df

if __name__ == "__main__":
    clean_water_dataset()
