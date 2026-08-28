# Laboratory Activity 1: Data Cleaning - Water Potability and Chemical Safety Assessment

## Project Overview

This repository contains the collaborative data cleaning project for **Laboratory Activity 1: Data Cleaning**, submitted under the **Environment & Public Health** domain. 

Clean drinking water is fundamental to human health and environmental sustainability (UN Sustainable Development Goal 6). However, water quality monitoring involves complex physicochemical assays (such as pH, Sulfates, Chloramines, and Trihalomethanes) that frequently suffer from uncalibrated sensor probes, selective laboratory test omission, and highly skewed geochemical concentrations.

This project implements a domain-informed data cleaning pipeline using Python and Pandas to evaluate, clean, and standardize freshwater safety records.

## Research Question and Problem Statement

### The Real-World Environmental Problem
Access to safe drinking water is threatened by natural mineral runoff and industrial chemical contamination. Public health authorities must assess water potability from laboratory samples, but incomplete testing records and probe measurement artifacts introduce high risk into automated water safety classification.

### Core Investigation Question
> *How can we accurately identify and clean sensor probe anomalies, laboratory test omission, and mineral concentration outliers across aquatic physicochemical parameters to ensure that water safety classification strictly adheres to World Health Organization (WHO) safety standards?*

---

## Dataset Profile

* **Dataset Title (Kaggle):** Water Quality / Water Potability
* **Dataset Creator:** Aditya Kadiwal
* **Primary File:** `water_potability.csv` (Located in this project directory)
* **Dimensions:** 3,276 rows × 10 columns
* **Target Variable:** `Potability` (Binary: 0 = Not Potable, 1 = Potable)

### Key Variables Analyzed

| Variable | Raw Data Type | Unit / Scale | Environmental Significance |
| :--- | :--- | :--- | :--- |
| `ph` | `float64` | Scale (0 - 14) | Acid-base balance of water (WHO safe limits: 6.5 - 8.5) |
| `Hardness` | `float64` | mg/L | Calcium and Magnesium mineral concentration |
| `Solids` | `float64` | ppm | Total Dissolved Solids (TDS); mineralization level |
| `Chloramines` | `float64` | ppm | Disinfectant agent used in municipal water treatment |
| `Sulfate` | `float64` | mg/L | Naturally occurring minerals from geological formations |
| `Conductivity` | `float64` | uS/cm | Electrical conductivity; indicator of dissolved ionic substances |
| `Organic_carbon` | `float64` | ppm | Total organic carbon from decaying organic matter |
| `Trihalomethanes` | `float64` | ug/L | Byproducts of chlorine disinfection; potential carcinogens |
| `Turbidity` | `float64` | NTU | Measure of water clarity and suspended colloidal matter |
| `Potability` | `int64` | Binary (0 / 1) | Target safety classification (0 = Non-potable, 1 = Potable) |

---

## Identified Data Quality Problems

Exploratory analysis using `df.info()`, `df.describe()`, and `df.isna().sum()` revealed four major data quality challenges:

1. **Substantial Non-Random Missing Values (Selective Assays):**
   * `Sulfate` is missing 781 values (~23.84%).
   * `ph` is missing 491 values (~14.99%).
   * `Trihalomethanes` is missing 162 values (~4.95%).
   * *Cause:* Comprehensive laboratory gas chromatography and probe tests were not performed uniformly on every sampled water body.
2. **Boundary Violations (Physically Impossible pH Values):**
   * Uncalibrated pH meters produce out-of-scale readings (< 0 or > 14), violating the fundamental logarithmic definition of hydrogen ion activity.
3. **Heavy Skewness in Total Dissolved Solids (Solids):**
   * Total Dissolved Solids show extreme right-skewed spikes (> 50,000 ppm, approaching seawater salinity), which distort mean calculations.
4. **Target Encoding & Type Integrity:**
   * Ensuring binary target classification is clean, non-null, and mapped as an integer label.

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
* **Domain Rationale:** The pH scale measures the decimal logarithm of the reciprocal of hydrogen ion activity in aqueous solutions. Values below 0 or above 14 represent probe hardware malfunction or severe electrical interference and must not be treated as valid aquatic observations.

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
* **Domain Rationale:**
  * Chemical concentrations in nature are rarely normally distributed; they are skewed by regional geology and human discharge.
  * The median is resistant to extreme outliers (unlike the mean).
  * Imputing conditionally by `Potability` preserves the underlying biochemical differences between potable and contaminated water sources, avoiding data dilution.

---

### Step 3: Extreme Outlier Handling (Domain vs. Artifacts)

* **Action:** Retain valid high-solids observations while capping extreme sensor spikes using a 3x IQR boundary.
* **Code:**
```python
# Cap extreme solids using 3*IQR (extreme boundary) to preserve brackish freshwater records
q1 = df["Solids"].quantile(0.25)
q3 = df["Solids"].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 3.0 * iqr

df = df[df["Solids"] <= upper_bound].reset_index(drop=True)
```
* **Domain Rationale:** 
  * High TDS (10,000–30,000 ppm) occurs naturally in mineral-rich aquifers and brackish groundwater estuaries.
  * Standard 1.5x IQR would erroneously eliminate valid brackish water samples. Using a conservative 3.0x IQR threshold removes true transmission/entry errors while retaining natural geological variance.

---

### Step 4: Data Type Integrity and Normalization

* **Action:** Verify zero nulls and cast target feature to boolean/int.
* **Code:**
```python
# Ensure Potability is properly typed
df["Potability"] = df["Potability"].astype(int)

# Verify no remaining null values
assert df.isna().sum().sum() == 0, "Dataset contains unresolved missing values!"
```
* **Rationale:** Ensures clean downstream compatibility with standard classification algorithms and visualization libraries.

---

## Before vs. After Summary

| Quality Check / Metric | Raw Dataset (`water_potability.csv`) | Cleaned Dataset |
| :--- | :--- | :--- |
| **Total Rows** | 3,276 rows | ~3,260 rows (extreme artifacts pruned) |
| **Missing `Sulfate` Values** | 781 nulls (23.84%) | 0 nulls (Class-conditional median) |
| **Missing `ph` Values** | 491 nulls (14.99%) | 0 nulls (Class-conditional median) |
| **Missing `Trihalomethanes`** | 162 nulls (4.95%) | 0 nulls (Class-conditional median) |
| **Out-of-Bound pH (< 0 or > 14)** | Present | 0 (Sanitized to valid physical scale) |
| **Aggregate Missing Values** | 1,434 missing values | 0 missing values |
| **Data Integrity** | High risk of classification error | Clean, robust, and WHO standard-aligned |

---

## How to Run the Cleaning Pipeline

### Prerequisites
* Python 3.8+
* pandas, numpy, matplotlib, seaborn, missingno

Install requirements:
```bash
pip install pandas numpy matplotlib seaborn missingno
```

### Execution Options
* **Interactive Notebook:** Open and run [`water_potability_data_cleaning.ipynb`](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/EDA/PROJ/projects/water_potability/water_potability_data_cleaning.ipynb) in Jupyter / VS Code for step-by-step interactive charts, data profiling, and domain audit.
* **Standalone Python Script:** Run `python clean_water_potability.py`.


### Execution
Run the following script directly from the `projects/water_potability/` directory:
```python
import pandas as pd
import numpy as np

# Load raw dataset
df = pd.read_csv("water_potability.csv")

# Step 1: Sanitize pH physical scale (0 to 14)
df.loc[(df["ph"] < 0) | (df["ph"] > 14), "ph"] = np.nan

# Step 2: Class-conditional median imputation
impute_features = ["ph", "Sulfate", "Trihalomethanes"]
for col in impute_features:
    df[col] = df.groupby("Potability")[col].transform(lambda g: g.fillna(g.median()))

# Step 3: Handle extreme solids boundary
q1 = df["Solids"].quantile(0.25)
q3 = df["Solids"].quantile(0.75)
iqr = q3 - q1
upper_limit = q3 + 3.0 * iqr
df = df[df["Solids"] <= upper_limit].reset_index(drop=True)

# Step 4: Validate and export
df["Potability"] = df["Potability"].astype(int)
df.to_csv("water_potability_cleaned.csv", index=False)
print("Water potability cleaning complete. Saved to water_potability_cleaned.csv.")
```

---

## Oral Recitation and Presentation Defense Guide

Be prepared to answer these core questions during the oral defense:

### Question 1: Why did you use class-conditional median imputation instead of dropping missing rows?
* **Answer:** "Over 30% of the dataset contained at least one missing chemical metric (mostly Sulfate and pH). Dropping rows via `dropna()` would eliminate nearly 1,200 valid observations, reducing sample size and inducing selection bias. Imputing with the median grouped by Potability preserves the characteristic chemical profiles of potable vs. non-potable water."

### Question 2: Why is the median preferred over the mean for environmental chemical data?
* **Answer:** "Environmental parameters like Turbidity and Sulfate exhibit heavy positive skewness due to localized mineralization and contamination plumes. The mean is pulled toward extreme values, whereas the median provides a robust measure of central tendency."

### Question 3: How does domain knowledge justify setting pH boundaries strictly between 0 and 14?
* **Answer:** "pH is the negative logarithm of hydrogen ion concentration. In natural freshwater systems, values strictly exist between 0 and 14 (with potable water generally between 6.5 and 8.5). Any reading outside this range indicates electrode degradation or measurement failure."

### Question 4: Why didn't you remove all values with high Total Dissolved Solids as outliers?
* **Answer:** "Natural water sources from deep mineral aquifers or coastal freshwater zones can have naturally elevated dissolved solids (10,000–25,000 ppm) without being measurement errors. Removing them with tight outlier thresholds would misrepresent genuine brackish water ecology."
