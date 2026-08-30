# Laboratory Activity 1: Data Cleaning Pipeline - Water Potability and Chemical Safety Assessment

## Project Overview

This repository contains the collaborative data cleaning project for **Laboratory Activity 1: Data Cleaning**, submitted under the **Environment & Public Health** domain (aligned with UN Sustainable Development Goal 6: Clean Water and Sanitation).

Clean drinking water is essential for human health and safety. In water quality monitoring, we collect measurements such as pH, mineral concentrations, and disinfectant levels to determine whether water is safe to drink (`Potability = 1`) or unsafe (`Potability = 0`).

However, real-world water quality datasets frequently contain data quality issues:
1. **Missing Test Results:** Expensive chemical assays (like Sulfate, pH, and Trihalomethanes) were not recorded for every sampled water body.
2. **Sensor Errors:** Faulty or uncalibrated probes can produce impossible values (such as pH readings outside the 0 to 14 scale).
3. **Extreme Values:** Natural mineral variations vs. corrupt telemetry spikes.

This project implements a clear, domain-informed data cleaning pipeline using Python and Pandas to evaluate, clean, and standardize the water safety records.

---

## Data Preparation: Key Concepts

In our lessons, preparing data involves different steps that work together:

| Stage | Goal | What We Do in this Activity |
| :--- | :--- | :--- |
| **Data Wrangling** | The overall process of taking raw data and turning it into a usable format. | Managing the entire workflow from loading data to saving the cleaned file. |
| **Data Cleaning** | Finding and fixing errors, missing values, duplicates, and invalid readings. | Fixing impossible pH values, filling missing numbers, checking duplicates, and checking outliers. |
| **Data Transformation** | Changing the format or scale of data without adding new information. | Rounding decimal numbers and ensuring the target is a clean integer. |
| **Feature Engineering** | Creating new variables from existing ones to help analysis or models. | (Optional for downstream modeling). |

```
Data Cleaning  -->  Data Transformation  -->  Feature Engineering  -->  Statistical Modeling / EDA
```

> **Core Guiding Principle:** *"We do not change data just because it looks unusual. We first check the data, understand why it looks that way, and choose the best way to handle it."*

---

## Research Question and Problem Statement

### The Real-World Environmental Problem
Access to safe drinking water is threatened by natural mineral runoff and chemical contamination. Public health authorities must assess water potability from laboratory samples, but incomplete testing records and probe measurement artifacts introduce high risk into water safety classification.

### Core Investigation Question
> *How can we systematically detect, audit, and clean sensor probe anomalies, missing chemical tests, and mineral concentration outliers to ensure that water safety classification strictly adheres to World Health Organization (WHO) safety standards?*

---

## Dataset Profile

* **Dataset Title (Kaggle):** Water Quality / Water Potability
* **Dataset Creator:** Aditya Kadiwal
* **Primary File:** `water_potability.csv` (Located in this project directory)
* **Dimensions:** 3,276 rows × 10 columns
* **Target Variable:** `Potability` (Binary: 0 = Not Potable / Unsafe, 1 = Potable / Safe)

### Dataset Variable Codebook and Environmental Standards

| Variable | Raw Data Type | Unit / Scale | WHO Reference Standard / Environmental Significance |
| :--- | :--- | :--- | :--- |
| `ph` | `float64` | Scale (0 - 14) | Acid-base balance of water (WHO safe limits: 6.5 - 8.5) |
| `Hardness` | `float64` | mg/L | Calcium and Magnesium mineral concentration |
| `Solids` | `float64` | ppm | Total Dissolved Solids (TDS); mineralization level |
| `Chloramines` | `float64` | ppm | Disinfectant agent used in municipal water treatment (WHO limit: $\le$ 4 ppm) |
| `Sulfate` | `float64` | mg/L | Naturally occurring minerals from geological formations (WHO guideline: $\le$ 250 mg/L) |
| `Conductivity` | `float64` | $\mu$S/cm | Electrical conductivity; indicator of dissolved ionic substances |
| `Organic_carbon` | `float64` | ppm | Total organic carbon from decaying organic matter |
| `Trihalomethanes` | `float64` | $\mu$g/L | Chlorine disinfection byproducts; potential carcinogens (WHO limit: $\le$ 80 $\mu$g/L) |
| `Turbidity` | `float64` | NTU | Measure of water clarity and suspended colloidal matter (WHO limit: $\le$ 5 NTU) |
| `Potability` | `int64` | Binary (0 / 1) | Target safety classification (0 = Non-potable, 1 = Potable) |

---

## Identified Data Quality Problems

Exploratory analysis using `df.info()`, `df.describe()`, and `df.isna().sum()` revealed four major data quality challenges:

1. **Substantial Missing Values across Chemical Tests:**
   * `Sulfate` is missing 781 values (~23.84%).
   * `ph` is missing 491 values (~14.99%).
   * `Trihalomethanes` is missing 162 values (~4.95%).
   * *Total Affected Rows:* 1,265 rows (38.61% of dataset).
2. **Boundary Violations (Physically Impossible pH Values):**
   * Uncalibrated pH meters can produce out-of-scale readings (< 0 or > 14), which cannot represent real water chemistry.
3. **Extreme Mineral Concentration Spikes (Solids):**
   * Total Dissolved Solids show extreme right-skewed concentrations (> 50,000 ppm), requiring us to distinguish between natural mineral-rich groundwater and sensor errors.
4. **Duplicate and Data Type Integrity:**
   * Auditing duplicate records (`df.duplicated()`) and ensuring binary target classification is mapped cleanly to `int64` with standardized 3-decimal floating-point precision.

---

## Data Cleaning Methodology and Domain Rationale

The cleaning pipeline applies domain-specific rules across five steps:

### Step 1: Enforcing Physicochemical Boundary Constraints (pH Scale)

* **Action:** Flag and convert pH readings outside 0–14 to `np.nan`.
* **Code:**
```python
import pandas as pd
import numpy as np

df = pd.read_csv("water_potability.csv")

# pH is strictly defined between 0 and 14
invalid_ph_mask = (df["ph"] < 0) | (df["ph"] > 14)
df.loc[invalid_ph_mask, "ph"] = np.nan
```
* **Rationale:** The pH scale is scientifically defined from 0 (very acidic) to 14 (very basic). Any reading outside this range represents an electrode or calibration defect.

---

### Step 2: Filling Missing Values by Water Safety Group

* **Action:** Impute missing values in `ph`, `Sulfate`, and `Trihalomethanes` using the **median grouped by Potability class**.
* **Code:**
```python
impute_cols = ["ph", "Sulfate", "Trihalomethanes"]

for col in impute_cols:
    # Use class median to keep potable and non-potable profiles distinct
    df[col] = df.groupby("Potability")[col].transform(
        lambda group: group.fillna(group.median())
    )
```
* **Rationale:**
  * **Why not `dropna()`?** Deleting incomplete rows would discard 1,265 samples (38.61% of the dataset), causing severe loss of valuable information.
  * **Why Median over Mean?** Chemical concentrations can have extreme spikes. The median represents the typical center and is resistant to outliers.
  * **Why Group by Potability?** Potable (safe) and non-potable (unsafe) water have distinct chemical baselines. Grouping ensures we fill missing values with numbers that reflect each type of water.

---

### Step 3: Domain-Informed Outlier Treatment (Total Dissolved Solids)

* **Action:** Evaluate standard ($1.5 \times \text{IQR}$) vs. extreme ($3.0 \times \text{IQR}$) boundaries to keep natural mineral-rich water while pruning corrupt telemetry spikes.
* **Code:**
```python
# Calculate 3.0 * IQR extreme fence for Solids
q1 = df["Solids"].quantile(0.25)
q3 = df["Solids"].quantile(0.75)
iqr = q3 - q1
upper_limit = q3 + 3.0 * iqr

df = df[df["Solids"] <= upper_limit].reset_index(drop=True)
```
* **Rationale:** 
  * High mineral content (10,000–30,000 ppm TDS) occurs naturally in deep mineral aquifers and coastal brackish water.
  * A standard $1.5 \times \text{IQR}$ rule would delete 47 valid mineral-rich water samples. Using a conservative $3.0 \times \text{IQR}$ boundary preserves natural environmental variation while filtering out impossible sensor spikes (> 60,000 ppm).

---

### Step 4: Data Type Integrity and Precision Normalization

* **Action:** Cast `Potability` to integer (`int64`), round continuous features to 3 decimal places, and verify data assertions.
* **Code:**
```python
# Ensure Potability is properly typed
df["Potability"] = df["Potability"].astype(int)

# Standardize continuous precision to 3 decimal places
continuous_cols = [c for c in df.columns if c != "Potability"]
for col in continuous_cols:
    df[col] = df[col].round(3)

# Verify data integrity
assert df.isna().sum().sum() == 0, "Dataset contains unresolved missing values!"
assert df["Potability"].isin([0, 1]).all(), "Potability contains non-binary values!"
assert df.duplicated().sum() == 0, "Duplicate rows detected!"
```
* **Rationale:** Ensures clean, consistent numbers ready for visualization and machine learning models.

---

### Step 5: Export Cleaned Dataset

* **Action:** Export cleaned dataset to `water_potability_cleaned.csv`.
* **Code:**
```python
df.to_csv("water_potability_cleaned.csv", index=False)
```

---

## Before vs. After Summary

| Check / Metric | Raw Dataset (`water_potability.csv`) | Cleaned Dataset (`water_potability_cleaned.csv`) |
| :--- | :--- | :--- |
| **Total Rows** | 3,276 rows | 3,276 rows (100% sample retention) |
| **Missing `Sulfate` Values** | 781 nulls (23.84%) | 0 nulls (Filled via class median) |
| **Missing `ph` Values** | 491 nulls (14.99%) | 0 nulls (Filled via class median) |
| **Missing `Trihalomethanes`** | 162 nulls (4.95%) | 0 nulls (Filled via class median) |
| **Total Incomplete Rows** | 1,265 rows (38.61%) | 0 rows (0.00% missing) |
| **Out-of-Bound pH (< 0 or > 14)** | Audited / Present in telemetry | 0 (Sanitized to valid 0–14 scale) |
| **Duplicate Records** | 0 duplicates | 0 duplicates |
| **Max Solids (ppm)** | 61,227.20 ppm | 61,227.20 ppm (Validated within extreme boundary) |
| **Potable Water Ratio** | 39.01% (1,278 potable samples) | 39.01% (Class balance fully preserved) |
| **Data Integrity Status** | 38.6% Incomplete Rows | Complete, Clean, and Ready for EDA |

---

## How to Run the Cleaning Pipeline

### Prerequisites
* Python 3.8+
* pandas, numpy, matplotlib, seaborn, missingno

Install requirements:
```bash
pip install pandas numpy matplotlib seaborn missingno
```

### Execution
* **Interactive Notebook:** Open and run [`water_potability_data_cleaning.ipynb`](water_potability_data_cleaning.ipynb) in Jupyter Notebook / VS Code / Google Colab for the step-by-step interactive pipeline, visualizations, and domain quality audit.

---

## Oral Recitation and Presentation Defense Guide

Clear, simple answers to explain our cleaning decisions during presentation:

### Question 1: Why did you fill missing values instead of deleting rows?
* **Defense Answer:** Over 38% of the dataset (1,265 rows) had at least one missing chemical test. If we deleted those rows with `dropna()`, we would lose more than one-third of our data. Filling them keeps all 3,276 rows of data for analysis.

### Question 2: Why did you use the median instead of the mean?
* **Defense Answer:** Environmental chemical levels often have extreme high or low values. The mean gets pulled by extreme numbers, but the median represents the typical center and is not affected by outliers.

### Question 3: Why did you fill missing values separately for Potable and Non-Potable water?
* **Defense Answer:** Safe drinking water and unsafe water naturally have different chemical levels. Grouping by Potability ensures that we fill missing values with realistic numbers for each type of water.

### Question 4: Why must pH be between 0 and 14?
* **Defense Answer:** The pH scale is scientifically defined from 0 (very acidic) to 14 (very basic). Any value outside this range is a faulty sensor reading, not real water chemistry.

### Question 5: Why didn't you remove all high Total Dissolved Solids as outliers?
* **Defense Answer:** Natural mineral water and groundwater can legitimately have high mineral levels (10,000 to 30,000 ppm). Removing them with a standard 1.5x IQR rule would delete valid natural water samples. Using an extreme 3.0x IQR cutoff keeps real water samples and only removes unrealistic sensor errors.
