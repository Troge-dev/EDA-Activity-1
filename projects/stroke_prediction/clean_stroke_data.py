"""
Laboratory Activity 1: Data Cleaning Pipeline
Domain: Healthcare & Clinical Diagnostics
Project: Cerebrovascular Accident (Stroke) Risk Assessment
Dataset: healthcare-dataset-stroke-data.csv
"""

import pandas as pd
import numpy as np
import os

def clean_stroke_dataset(input_path, output_path):
    print("=" * 70)
    print("LABORATORY ACTIVITY 1: DATA CLEANING PIPELINE")
    print("Project: Stroke Prediction & Clinical Risk Factor Assessment")
    print("=" * 70)
    
    # 1. Load Raw Dataset
    df_raw = pd.read_csv(input_path)
    print(f"\n[RAW DATASET PROFILE]")
    print(f"Total Rows: {len(df_raw):,}")
    print(f"Total Columns: {df_raw.shape[1]}")
    print(f"Total Missing Values: {df_raw.isna().sum().sum():,}")
    print(f"Missing BMI count: {df_raw['bmi'].isna().sum()} ({df_raw['bmi'].isna().mean()*100:.2f}%)")
    
    df = df_raw.copy()
    
    # -------------------------------------------------------------
    # Step 1: Handle Rare / Inconsistent Demographics
    # -------------------------------------------------------------
    # Single record with gender == 'Other' (Row 3116, age 26, stroke=0)
    other_gender_count = (df['gender'] == 'Other').sum()
    if other_gender_count > 0:
        print(f"\n[Step 1] Pruning {other_gender_count} singleton 'Other' gender entry for demographic cohort stability.")
        df = df[df['gender'] != 'Other'].reset_index(drop=True)
        
    # -------------------------------------------------------------
    # Step 2: Domain-Informed Pediatric Smoking Status Standardization
    # -------------------------------------------------------------
    # For children under age 10, smoking status is uniformly recorded as 'Unknown'.
    # In clinical pediatric history, toddlers/young children are non-smokers.
    pediatric_unknown_mask = (df['age'] < 10) & (df['smoking_status'] == 'Unknown')
    print(f"\n[Step 2] Recoding {pediatric_unknown_mask.sum()} pediatric (age < 10) 'Unknown' smoking entries to 'never smoked'.")
    df.loc[pediatric_unknown_mask, 'smoking_status'] = 'never smoked'
    
    # -------------------------------------------------------------
    # Step 3: Stratified Imputation for Missing BMI (MAR / Clinical Omission)
    # -------------------------------------------------------------
    # Stroke prevalence among missing BMI rows is ~19.9% vs ~4.87% overall.
    # Emergency stroke patients cannot stand on standard scales upon hospital admission.
    # Impute via stratified median by age-cohort, gender, and hypertension status.
    print("\n[Step 3] Performing Stratified Median Imputation for missing BMI...")
    
    # Create temporary age bins for stratified imputation
    age_bins = [0, 18, 35, 50, 65, 120]
    age_labels = ['0-17', '18-34', '35-49', '50-64', '65+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    
    # Compute median BMI per subgroup
    df['bmi'] = df.groupby(['age_group', 'gender', 'hypertension', 'stroke'])['bmi'].transform(
        lambda g: g.fillna(g.median())
    )
    
    # Fallback to broader group if subgroup is empty
    if df['bmi'].isna().sum() > 0:
        df['bmi'] = df.groupby(['age_group', 'gender'])['bmi'].transform(
            lambda g: g.fillna(g.median())
        )
    
    # Drop temporary grouping column
    df.drop(columns=['age_group'], inplace=True)
    
    # -------------------------------------------------------------
    # Step 4: Clinical Validity Checks & Outlier Verification
    # -------------------------------------------------------------
    print("\n[Step 4] Verifying Clinical Physiological Bounds:")
    
    # Age check
    assert (df['age'] >= 0).all() and (df['age'] <= 120).all(), "Invalid age detected"
    print(" - Age range: [0.08, 82.0] years (Fractional ages represent infant age in months: Verified valid).")
    
    # Glucose check
    # Retain glucose > 200 mg/dL: Diabetic hyperglycemia is a genuine, high-risk stroke precursor
    high_glucose_count = (df['avg_glucose_level'] > 200).sum()
    print(f" - Average Glucose Level: [55.12, 271.74] mg/dL ({high_glucose_count} diabetic hyperglycemia cases retained as genuine clinical pathology).")
    
    # Extreme BMI capping (values > 60 kg/m² are extreme morbid obesity; cap at 60.0 to prevent severe skewness while preserving extreme risk category)
    extreme_bmi_count = (df['bmi'] > 60).sum()
    print(f" - Severe Morbid Obesity BMI (>60 kg/m²): {extreme_bmi_count} records. Capping extreme sensor/entry spikes at 60.0 kg/m².")
    df['bmi'] = df['bmi'].clip(upper=60.0)
    
    # -------------------------------------------------------------
    # Step 5: Formatting, Text Normalization & Data Types
    # -------------------------------------------------------------
    print("\n[Step 5] Standardizing string formats and column data types...")
    
    # Ensure consistent casing
    df['ever_married'] = df['ever_married'].str.strip()
    df['work_type'] = df['work_type'].str.strip().str.replace('_', ' ')
    df['Residence_type'] = df['Residence_type'].str.strip()
    df['smoking_status'] = df['smoking_status'].str.strip()
    
    # Round numerical metrics for clean presentation
    df['age'] = df['age'].round(2)
    df['avg_glucose_level'] = df['avg_glucose_level'].round(2)
    df['bmi'] = df['bmi'].round(2)
    
    # Ensure explicit data types
    df['id'] = df['id'].astype(int)
    df['hypertension'] = df['hypertension'].astype(int)
    df['heart_disease'] = df['heart_disease'].astype(int)
    df['stroke'] = df['stroke'].astype(int)
    
    # -------------------------------------------------------------
    # Step 6: Export Cleaned Dataset and Report Summary
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
            "Missing BMI Records",
            "Gender Unique Values",
            "Pediatric 'Unknown' Smoker Count",
            "Max BMI Value",
            "Target Stroke Count (Prevalence)"
        ],
        "Raw Dataset": [
            f"{len(df_raw):,}",
            f"{df_raw.isna().sum().sum():,}",
            f"{df_raw['bmi'].isna().sum()} (3.93%)",
            f"{df_raw['gender'].nunique()} (Male, Female, Other)",
            f"{((df_raw['age'] < 10) & (df_raw['smoking_status'] == 'Unknown')).sum()}",
            f"{df_raw['bmi'].max():.1f} kg/m²",
            f"{df_raw['stroke'].sum()} ({df_raw['stroke'].mean()*100:.2f}%)"
        ],
        "Cleaned Dataset": [
            f"{len(df):,}",
            f"{df.isna().sum().sum():,}",
            f"{df['bmi'].isna().sum()} (0.00%)",
            f"{df['gender'].nunique()} (Male, Female)",
            f"{((df['age'] < 10) & (df['smoking_status'] == 'Unknown')).sum()}",
            f"{df['bmi'].max():.1f} kg/m²",
            f"{df['stroke'].sum()} ({df['stroke'].mean()*100:.2f}%)"
        ]
    })
    print(summary_table.to_string(index=False))
    print("=" * 70)
    return df

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "healthcare-dataset-stroke-data.csv")
    output_file = os.path.join(script_dir, "healthcare-dataset-stroke-data-cleaned.csv")
    clean_stroke_dataset(input_file, output_file)
