# Laboratory Activity 1: Data Cleaning Pipeline - Water Potability and Physicochemical Safety Assessment

## Project Overview

This repository contains the collaborative data cleaning project for **Laboratory Activity 1: Data Cleaning**, submitted under the **Environment & Public Health** domain (aligned with UN Sustainable Development Goal 6: Clean Water and Sanitation).

Clean drinking water is fundamental to human health and environmental biosecurity. However, water quality monitoring involves complex physicochemical assays (such as pH, Sulfates, Chloramines, and Trihalomethanes) that frequently suffer from uncalibrated sensor probes, selective laboratory test omission, and highly skewed geochemical concentrations.

This project implements a domain-informed data cleaning pipeline using Python and Pandas to evaluate, clean, and standardize freshwater safety records.

---

## Theoretical Framework: Data Preparation Lifecycle

In data science and exploratory data analysis (DS311), preparing data for analysis involves distinct disciplines that operate in sequence:

| Stage | Core Objective | Scope in this Activity |
| :--- | :--- | :--- |
| **Data Wrangling** | The broad, end-to-end umbrella process of gathering, structuring, cleaning, enriching, validating, and publishing data. | Overall workflow management from raw CSV ingestion to reporting. |
| **Data Cleaning** | Detecting and correcting (or removing) corrupt, inaccurate, incomplete, duplicate, or out-of-bound records. | Enforcing pH boundaries (0–14), handling missing values via domain imputation, auditing duplicates, and pruning sensor artifacts. |
| **Data Transformation** | Modifying the form, scale, or representation of variables without introducing new conceptual information. | Standardizing floating-point precision and validating target encoding format. |
| **Feature Engineering** | Creating new domain-specific variables to improve predictive modeling. | Downstream modeling preparation (e.g., composite mineral ratios). |

```
Data Cleaning  -->  Data Transformation  -->  Feature Engineering  -->  Statistical Modeling / EDA
```

> **Core Guiding Principle:** *"We should not change data simply because it looks unusual. We first identify the problem, investigate its root cause and domain context, and then decide how it should be rigorously handled."*

---

## Research Question and Problem Statement

### The Real-World Environmental Problem
Access to safe drinking water is threatened by natural mineral runoff and industrial chemical contamination. Public health authorities must assess water potability from laboratory samples, but incomplete testing records and probe measurement artifacts introduce high risk into automated water safety classification.

### Core Investigation Question
> *How can we systematically detect, audit, and clean sensor probe anomalies, laboratory test omissions, and mineral concentration outliers across aquatic physicochemical parameters to ensure that water safety classification strictly adheres to World Health Organization (WHO) safety standards?*

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

## Identified Data Quality Problems and Missingness Taxonomy

Exploratory analysis using `df.info()`, `df.describe()`, and `df.isna().sum()` revealed four major data quality challenges evaluated under **Rubin's Missing Data Taxonomy (Rubin, 1976; Little & Rubin, 2020)**:

1. **Substantial Missing Values under Missing at Random (MAR):**
   * `Sulfate` is missing 781 values (~23.84%).
   * `ph` is missing 491 values (~14.99%).
   * `Trihalomethanes` is missing 162 values (~4.95%).
   * *Total Affected Rows:* 1,265 rows (38.61% of dataset).
   * *Mechanism:* Missingness is related to selective laboratory assay protocols across sampled water bodies rather than completely random omission (MCAR).
2. **Boundary Violations (Physically Impossible pH Values):**
   * Uncalibrated pH meters produce out-of-scale readings (< 0 or > 14), violating the fundamental logarithmic definition of hydrogen ion activity ($-\log_{10}[a_{\text{H}^+}]$).
3. **Heavy Skewness in Total Dissolved Solids (Solids):**
   * Total Dissolved Solids show extreme right-skewed concentrations (> 50,000 ppm, approaching seawater salinity), requiring domain-informed distinction between natural brackish variance and sensor telemetry errors.
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
* **Domain Rationale:** The pH scale measures the negative decimal logarithm of hydrogen ion activity in aqueous solutions. Values below 0 or above 14 represent probe hardware malfunction or calibration drift and must not be treated as valid aquatic observations.

---

### Step 2: Class-Conditional Median Imputation for Missing Chemical Metrics

* **Action:** Impute missing values in `ph`, `Sulfate`, and `Trihalomethanes` using the **median grouped by Potability class**.
* **Code:**
```python
impute_cols = ["ph", "Sulfate", "Trihalomethanes"]

for col in impute_cols:
    # Use class-conditional median to prevent cross-class contamination
    df[col] = df.groupby("Potability")[col].transform(
        lambda group: group.fillna(group.median())
    )
```
* **Domain & Statistical Rationale:**
  * **Why Not `dropna()`?** Dropping rows would eliminate 1,265 observations (38.61% of data), severely depleting statistical power and inducing selection bias.
  * **Why Median Over Mean?** Environmental parameters are skewed by localized mineralization. The median provides an outlier-resistant measure of central tendency.
  * **Why Class-Conditional?** Potable and non-potable water exhibit distinct geochemical baselines. Grouping by `Potability` preserves class-specific profiles under the MAR framework.

---

### Step 3: Domain-Informed Outlier Treatment (Total Dissolved Solids)

* **Action:** Evaluate mild ($1.5 \times \text{IQR}$) vs. extreme ($3.0 \times \text{IQR}$) fences to retain valid mineral-rich groundwater while capping corrupt telemetry spikes.
* **Code:**
```python
# Calculate 3.0 * IQR extreme fence for Solids
q1 = df["Solids"].quantile(0.25)
q3 = df["Solids"].quantile(0.75)
iqr = q3 - q1
upper_limit = q3 + 3.0 * iqr

df = df[df["Solids"] <= upper_limit].reset_index(drop=True)
```
* **Domain Rationale:** 
  * High TDS (10,000–30,000 ppm) occurs naturally in deep mineral aquifers and brackish estuaries.
  * Standard $1.5 \times \text{IQR}$ would erroneously eliminate 47 valid mineral-rich water samples (1.4%). Using a conservative $3.0 \times \text{IQR}$ threshold removes true telemetry errors while preserving genuine ecological variability.

---

### Step 4: Data Type Integrity and Precision Normalization

* **Action:** Enforce integer typing on `Potability`, round continuous features to 3 decimal places, and verify pipeline assertions.
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
* **Rationale:** Ensures numerical stability and seamless compatibility with machine learning classifiers and EDA plotting routines.

---

### Step 5: Export Cleaned Dataset

* **Action:** Export cleaned dataset to `water_potability_cleaned.csv`.
* **Code:**
```python
df.to_csv("water_potability_cleaned.csv", index=False)
```

---

## Before vs. After Summary

| Quality Check / Metric | Raw Dataset (`water_potability.csv`) | Cleaned Dataset (`water_potability_cleaned.csv`) |
| :--- | :--- | :--- |
| **Total Rows** | 3,276 rows | 3,276 rows (100% sample retention) |
| **Missing `Sulfate` Values** | 781 nulls (23.84%) | 0 nulls (Class-conditional median) |
| **Missing `ph` Values** | 491 nulls (14.99%) | 0 nulls (Class-conditional median) |
| **Missing `Trihalomethanes`** | 162 nulls (4.95%) | 0 nulls (Class-conditional median) |
| **Total Incomplete Rows** | 1,265 rows (38.61%) | 0 rows (0.00% missing) |
| **Out-of-Bound pH (< 0 or > 14)** | Audited / Present in telemetry | 0 (Sanitized to valid physical scale) |
| **Duplicate Records** | 0 duplicates | 0 duplicates |
| **Max Solids (ppm)** | 61,227.20 ppm | 61,227.20 ppm (Validated within extreme fence) |
| **Potable Water Ratio** | 39.01% (1,278 potable samples) | 39.01% (Class balance fully preserved) |
| **Data Integrity Status** | Unsanitized / High null rate | Fully Cleaned / WHO-Aligned / Ready for EDA |

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

Be prepared to answer these core questions during the oral defense:

### Question 1: Why did you use class-conditional median imputation instead of dropping missing rows?
* **Defense Answer:** Over 38% of the dataset contained at least one missing chemical metric (`Sulfate` ~23.8%, `ph` ~15.0%, `Trihalomethanes` ~4.95%). Applying listwise deletion (`dropna()`) would eliminate 1,265 valid observations, significantly reducing statistical power and introducing severe selection bias (since missingness reflects selective testing rather than random failure). Grouping by `Potability` preserves the characteristic chemical signatures of potable vs. non-potable water under the Missing at Random (MAR) framework without introducing cross-class data leakage.

### Question 2: Why is the median preferred over the mean for aquatic chemical metrics?
* **Defense Answer:** Environmental chemical concentrations are heavily skewed by regional geological mineral deposits and industrial discharge plumes. The arithmetic mean is sensitive to extreme values, whereas the median provides a robust, outlier-resistant measure of central tendency that preserves the natural distribution shape.

### Question 3: How does domain knowledge justify setting pH boundaries strictly between 0 and 14?
* **Defense Answer:** pH is fundamentally defined as the negative logarithm of hydrogen ion activity ($-\log_{10}[a_{\text{H}^+}]$). In natural aqueous systems, valid readings strictly reside between 0 and 14 (potable drinking water is regulated by the WHO between 6.5 and 8.5). Values outside this range represent sensor hardware defects or calibration errors rather than actual water chemistry.

### Question 4: Why didn't you remove high Total Dissolved Solids using standard 1.5 x IQR?
* **Defense Answer:** Natural water sources from deep mineral aquifers or coastal brackish estuaries naturally reach 10,000 to 30,000 ppm TDS without being recording errors. As taught in our lesson, *we should not change data simply because it looks unusual*. A strict 1.5 x IQR filter would erroneously delete 47 valid mineral-rich water samples. Using a conservative 3.0 x IQR threshold retains genuine ecological variability while protecting against true sensor telemetry spikes.

### Question 5: What is the distinction between Data Cleaning and Data Transformation in your pipeline?
* **Defense Answer:** Data cleaning focused on detecting and fixing errors—enforcing pH physical boundaries, imputing missing assays, auditing duplicate records, and validating outlier limits. Data transformation standardized the representation without altering the underlying meaning—rounding continuous metrics to 3 decimal places for precision and explicitly casting the target `Potability` to `int64`.
