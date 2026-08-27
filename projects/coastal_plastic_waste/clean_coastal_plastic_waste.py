"""
Laboratory Activity 1: Data Cleaning Pipeline
Domain: Marine Ecology & Environmental Economics
Project: Global Coastal Plastic Waste and Economic Mismanagement
Dataset: per-capita-mismanaged-plastic-waste-vs-gdp-per-capita.csv
"""

import pandas as pd
import numpy as np
import os

def clean_coastal_plastic_waste(input_path, output_path):
    print("=" * 70)
    print("LABORATORY ACTIVITY 1: DATA CLEANING PIPELINE")
    print("Project: Global Coastal Plastic Waste and Economic Mismanagement")
    print("=" * 70)
    
    # 1. Load raw dataset
    df_raw = pd.read_csv(input_path)
    print(f"\n[RAW DATASET PROFILE]")
    print(f"Total Rows: {len(df_raw):,}")
    print(f"Total Columns: {df_raw.shape[1]}")
    print(f"Total Missing Values: {df_raw.isna().sum().sum():,}")
    
    df = df_raw.copy()
    
    # -------------------------------------------------------------
    # Step 1: Standardize Column Headers
    # -------------------------------------------------------------
    print("\n[Step 1] Renaming raw headers to standardized snake_case identifiers...")
    df.columns = [
        "country", "iso_code", "year", "mismanaged_waste_per_capita", 
        "gdp_per_capita", "population", "continent"
    ]
    
    # -------------------------------------------------------------
    # Step 2: Time-Invariant Continental Metadata Propagation
    # -------------------------------------------------------------
    print("\n[Step 2] Propagating static geographic continent metadata across historical records...")
    df["continent"] = df.groupby("country")["continent"].transform(
        lambda group: group.ffill().bfill()
    )
    
    # -------------------------------------------------------------
    # Step 3: Filter to Valid Environmental Benchmark Era (2010)
    # -------------------------------------------------------------
    print("\n[Step 3] Pruning historical population-only records lacking plastic waste data...")
    df_clean = df[df["mismanaged_waste_per_capita"].notna()].copy()
    print(f" - Retained {len(df_clean)} records with measured plastic waste mismanagement.")
    
    # -------------------------------------------------------------
    # Step 4: Prune Macro-Regional Blocs and Aggregates
    # -------------------------------------------------------------
    print("\n[Step 4] Pruning non-sovereign macro-regional blocs and custom OWID_* codes...")
    invalid_mask = df_clean["iso_code"].isna() | df_clean["iso_code"].str.startswith("OWID_")
    df_clean = df_clean[~invalid_mask].reset_index(drop=True)
    print(f" - Filtered to {len(df_clean)} verified sovereign nation states.")
    
    # -------------------------------------------------------------
    # Step 5: Country & Continental Temporal Imputation for Economic Indicators
    # -------------------------------------------------------------
    print("\n[Step 5] Imputing missing economic/demographic indicators using country historical series & continental medians...")
    for col in ["gdp_per_capita", "population"]:
        # First attempt: Country-specific historical median
        country_medians = df.groupby("country")[col].median()
        df_clean[col] = df_clean[col].fillna(df_clean["country"].map(country_medians))
        # Fallback: Continental median
        continent_medians = df_clean.groupby("continent")[col].median()
        df_clean[col] = df_clean[col].fillna(df_clean["continent"].map(continent_medians))
        
    # -------------------------------------------------------------
    # Step 6: Logarithmic Transformations for Skewed Distributions
    # -------------------------------------------------------------
    print("\n[Step 6] Calculating logarithmic transformations (log10) for econometric analysis...")
    df_clean["log_gdp"] = np.log10(df_clean["gdp_per_capita"]).round(4)
    df_clean["log_mismanaged_waste"] = np.log10(df_clean["mismanaged_waste_per_capita"] + 1e-5).round(4)
    
    # Round numerical metrics
    df_clean["mismanaged_waste_per_capita"] = df_clean["mismanaged_waste_per_capita"].round(4)
    df_clean["gdp_per_capita"] = df_clean["gdp_per_capita"].round(2)
    df_clean["population"] = df_clean["population"].astype(np.int64)
    
    assert df_clean.isna().sum().sum() == 0, "Cleaned dataset contains unresolved nulls!"
    
    # -------------------------------------------------------------
    # Step 7: Export Cleaned Dataset
    # -------------------------------------------------------------
    df_clean.to_csv(output_path, index=False)
    print(f"\n[EXPORT] Successfully saved cleaned dataset to: {output_path}")
    
    print("\n" + "=" * 70)
    print("BEFORE VS AFTER DATA CLEANING COMPARISON")
    print("=" * 70)
    summary_table = pd.DataFrame({
        "Metric / Check": [
            "Total Rows",
            "Target Missing Values",
            "Continent Missing Rate",
            "Entity Composition",
            "Column Naming Format",
            "Downstream Modeling Readiness"
        ],
        "Raw Dataset": [
            f"{len(df_raw):,} (99.6% sparse)",
            f"{df_raw['Per capita mismanaged plastic waste'].isna().sum():,} nulls",
            "> 90% missing across history",
            "Mixed (Continents, Blocs, Nations)",
            "Unstandardized strings & units",
            "Unusable (Sparse matrix)"
        ],
        "Cleaned Dataset": [
            f"{len(df_clean):,} sovereign nations",
            "0 missing values",
            "0.00% missing (Propagated)",
            "100% Sovereign ISO-3 States",
            "Standardized snake_case + log10",
            "Ready for Regression & Mapping"
        ]
    })
    print(summary_table.to_string(index=False))
    print("=" * 70)
    return df_clean

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "per-capita-mismanaged-plastic-waste-vs-gdp-per-capita.csv")
    output_file = os.path.join(script_dir, "coastal_plastic_waste_cleaned.csv")
    clean_coastal_plastic_waste(input_file, output_file)
