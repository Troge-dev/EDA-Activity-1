# Laboratory Activity 1: Data Cleaning - Stroke Prediction and Clinical Risk Factor Assessment

## Project Overview

This repository contains the collaborative data cleaning project for **Laboratory Activity 1: Data Cleaning**, submitted under the **Healthcare & Clinical Diagnostics** domain.

Cerebrovascular accidents (strokes) represent the second leading cause of mortality globally and a major contributor to long-term physical disability (World Health Organization). Hospital electronic health records (EHR) and clinical registries track key demographic, behavioral, and biochemical biomarkers (such as age, hypertension, heart disease, average blood glucose levels, and body mass index). However, clinical datasets frequently present data quality problems including missing physiological measurements due to acute emergency admissions, demographic singleton anomalies, structural missingness in pediatric histories, and extreme metabolic outliers.

This project designs and implements a domain-informed data cleaning pipeline using Python and Pandas to evaluate, clean, standardize, and prepare patient records for clinical analysis without introducing selection bias.

---

## Research Question and Problem Statement

### The Real-World Clinical Problem
Accurate early assessment of stroke vulnerability enables preventative cardiovascular interventions. However, clinical registries are vulnerable to systematic data-recording gaps. In acute emergency stroke admissions, incapacitated patients cannot undergo standard anthropometric measurements (such as standing on a scale for BMI), resulting in selective omission of critical data. Naively dropping incomplete patient records deletes high-risk clinical events, biasing epidemiological evaluations and compromising predictive accuracy.

### Core Investigation Question
> *How can we systematically detect and resolve missing anthropometric readings, structural survey artifacts in pediatric histories, and extreme metabolic glucose spikes in clinical health records to produce an accurate, bias-free dataset for stroke risk assessment?*

---

## Dataset Profile

* **Dataset Title (Kaggle):** [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
* **Dataset Creator:** fedesoriano (Source: Open Health Data Archive)
* **Primary File:** `healthcare-dataset-stroke-data.csv` (Located in this project directory)
* **Cleaned File:** `healthcare-dataset-stroke-data-cleaned.csv`
* **Dimensions:** 5,110 rows × 12 columns
* **Target Variable:** `stroke` (Binary: 0 = No stroke event, 1 = Stroke event)

### Key Variables Analyzed

| Variable | Raw Type | Unit / Scale | Clinical Significance |
| :--- | :--- | :--- | :--- |
| `id` | `int64` | Discrete ID | Unique patient identification number |
| `gender` | `object` | Nominal | Patient biological sex (`Male`, `Female`, `Other`) |
| `age` | `float64` | Years (0.08–82.0) | Patient age (fractional values represent infant age in months) |
| `hypertension` | `int64` | Binary (0 / 1) | Diagnosed high blood pressure (chronic vascular strain) |
| `heart_disease` | `int64` | Binary (0 / 1) | History of coronary artery disease / cardiac conditions |
| `ever_married` | `object` | Binary (`Yes`/`No`) | Marital status indicator (social demographic factor) |
| `work_type` | `object` | Nominal | Employment category (`Private`, `Self-employed`, `Govt_job`, `children`, `Never_worked`) |
| `Residence_type` | `object` | Nominal | Geographic living environment (`Urban`, `Rural`) |
| `avg_glucose_level`| `float64` | mg/dL | Serum fasting/postprandial glucose level |
| `bmi` | `float64` | kg/m² | Body Mass Index (weight in kg / height in m²) |
| `smoking_status` | `object` | Categorical | Smoking history (`formerly smoked`, `never smoked`, `smokes`, `Unknown`) |
| `stroke` | `int64` | Binary (0 / 1) | Target event: Diagnosed acute cerebrovascular accident |

---

## Identified Data Quality Problems

Exploratory analysis using `df.info()`, `df.describe()`, and `df.isna().sum()` revealed four major data quality challenges:

1. **Non-Random Missing Anthropometric Data (Missing BMI Mechanism):**
   * `bmi` is missing in **201 patients (~3.93% of the dataset)**.
   * *Critical Finding:* Among patients with missing BMI, the stroke prevalence is **19.90% (40 out of 201 patients)**, compared to only **4.87% in the general dataset population**.
   * *Root Cause:* This is a classic **Missing At Random (MAR) / Missing Not At Random (MNAR)** scenario in emergency medical registries. Patients arriving in acute stroke crises (e.g., severe hemiplegia, altered consciousness, or comatose states) cannot stand on standard medical scales during triage. Deleting missing rows via `dropna()` would discard **16.06% of all stroke cases (40 out of 249)**, introducing severe survival/triage bias.

2. **Pediatric Structural Missingness in Survey Fields (`smoking_status`):**
   * 1,544 rows contain `smoking_status == "Unknown"`.
   * For pediatric patients under age 10 (`age < 10`), **100% of records (472/472)** are coded as `"Unknown"`.
   * *Root Cause:* Pediatric intakes intentionally omit tobacco history questions for infants and young children. Treating children's `"Unknown"` status the same as adult unrecorded tobacco use distorts behavioral modeling.

3. **Singleton Rare Class in Demographic Attribute (`gender`):**
   * A single record contains `gender == "Other"` (Row index 3116, age 26, stroke=0).
   * *Root Cause:* Single-observation categories cause cross-validation folds and stratified sampling splits to fail or produce unstable demographic subgroup medians.

4. **Extreme Physiological Values & Outlier Interpretation:**
   * `avg_glucose_level` reaches 271.74 mg/dL (434 patients exhibit glucose > 200 mg/dL).
   * `bmi` ranges up to 97.6 kg/m² (13 patients have BMI > 60 kg/m²).
   * *Root Cause:* Glucose spikes represent genuine diabetic hyperglycemia rather than sensor artifacts, while extreme BMIs (> 60 kg/m²) represent severe morbid obesity combined with occasional manual data-entry scale errors.

---

## Data Cleaning Methodology and Domain Rationale

The cleaning pipeline applies domain-specific rules across five distinct steps:

### Step 1: Demographic Homogenization (Pruning Singleton Category)
* **Action:** Remove the single `"Other"` gender record to ensure demographic cohort stability during stratified subgroup analyses.
* **Code:**
```python
# Remove single 'Other' record to preserve clean binary demographic stratification
df = df[df["gender"] != "Other"].reset_index(drop=True)
```
* **Domain Rationale:** Statistical stratification requires adequate sample size per category. A single observation cannot support subgroup variance estimation or stratified imputation.

---

### Step 2: Domain-Informed Pediatric Recoding for `smoking_status`
* **Action:** Recode `smoking_status` from `"Unknown"` to `"never smoked"` for all pediatric patients (`age < 10`).
* **Code:**
```python
# Pediatric patients under 10 are clinically non-smokers
pediatric_mask = (df["age"] < 10) & (df["smoking_status"] == "Unknown")
df.loc[pediatric_mask, "smoking_status"] = "never smoked"
```
* **Domain Rationale:** In pediatric clinical documentation, clinicians do not interview infants and young children regarding tobacco usage. Their `"Unknown"` entries represent structural unapplicability rather than missing behavioral habits. Recoding them to `"never smoked"` aligns with epidemiological reality and clarifies the true adult `"Unknown"` subset.

---

### Step 3: Stratified Cohort Imputation for Missing BMI
* **Action:** Impute missing BMI using the **median of stratified clinical cohorts** grouped by `age_group`, `gender`, `hypertension`, and `stroke` status.
* **Code:**
```python
# Age stratification bins
age_bins = [0, 18, 35, 50, 65, 120]
age_labels = ["0-17", "18-34", "35-49", "50-64", "65+"]
df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False)

# Stratified median imputation preserving clinical risk profiles
df["bmi"] = df.groupby(["age_group", "gender", "hypertension", "stroke"])["bmi"].transform(
    lambda group: group.fillna(group.median())
)

df.drop(columns=["age_group"], inplace=True)
```
* **Domain Rationale:** 
  * Simple global mean/median imputation flattens biological diversity and ignores that elderly, hypertensive, and stroke-positive patients have systematically distinct metabolic profiles.
  * Stratifying by age bracket, biological sex, hypertension status, and stroke outcome ensures that imputed BMI values reflect the patient's specific physiological risk cohort without introducing data leakage across unstratified populations.

---

### Step 4: Metabolic Outlier Justification and Physical Scale Bounds
* **Action:** Retain elevated blood glucose levels (> 200 mg/dL); cap extreme BMI spikes at 60.0 kg/m².
* **Code:**
```python
# Validate age range (fractional ages < 1 represent infants in months)
assert (df["age"] >= 0).all() and (df["age"] <= 120).all()

# Retain glucose > 200 mg/dL: Diabetic hyperglycemia is a genuine stroke risk factor
# Cap extreme morbid obesity BMI at 60.0 kg/m² to control severe distortion
df["bmi"] = df["bmi"].clip(upper=60.0)
```
* **Domain Rationale:**
  * **Glucose:** In endocrinology, fasting glucose > 126 mg/dL indicates diabetes, and postprandial levels exceeding 200–270 mg/dL represent acute uncontrolled hyperglycemia. High glucose damages vascular endothelium, accelerating atherosclerosis. **These are genuine medical risk states, not measurement errors, and must be preserved.**
  * **BMI:** While Class III severe obesity is clinically defined at BMI ≥ 40 kg/m², values above 60.0 kg/m² are extremely rare and often reflect typing errors (e.g. height in inches entered into centimeter fields). Capping at 60.0 kg/m² preserves the severe obesity risk classification while restraining extreme leverage in statistical models.

---

### Step 5: Data Type Normalization and Formatting
* **Action:** Clean text formatting, normalize string categories, round continuous values, and enforce strict integer types on IDs and binary flags.
* **Code:**
```python
df["work_type"] = df["work_type"].str.strip().str.replace("_", " ")
df["Residence_type"] = df["Residence_type"].str.strip()
df["smoking_status"] = df["smoking_status"].str.strip()

# Enforce integer typing
for col in ["id", "hypertension", "heart_disease", "stroke"]:
    df[col] = df[col].astype(int)

# Round continuous features
df["age"] = df["age"].round(2)
df["avg_glucose_level"] = df["avg_glucose_level"].round(2)
df["bmi"] = df["bmi"].round(2)
```

---

## Before vs. After Summary

| Quality Metric / Check | Raw Dataset (`healthcare-dataset-stroke-data.csv`) | Cleaned Dataset (`healthcare-dataset-stroke-data-cleaned.csv`) |
| :--- | :--- | :--- |
| **Total Rows** | 5,110 rows | 5,109 rows (1 singleton anomaly pruned) |
| **Total Missing Values** | 201 missing values | 0 missing values |
| **Missing BMI Records** | 201 nulls (3.93%) | 0 nulls (Stratified cohort median imputed) |
| **Gender Categories** | 3 (`Male`, `Female`, `Other`) | 2 (`Male`, `Female`) |
| **Pediatric 'Unknown' Smokers** | 472 ambiguous child entries | 0 (Accurately recoded to `never smoked`) |
| **Maximum BMI Value** | 97.6 kg/m² (Severe unclipped spike) | 60.0 kg/m² (Capped at physiological boundary) |
| **Stroke Events Preserved** | 249 stroke cases (4.87%) | 249 stroke cases (4.87% - **100% preserved**) |
| **Data Integrity Level** | High risk of triage & survival bias | Statistically robust, clinically validated |

---

## How to Run the Cleaning Pipeline

### Prerequisites
```bash
pip install pandas numpy
```

### Execution
Run the standalone pipeline script from the project directory:
```bash
python projects/stroke_prediction/clean_stroke_data.py
```

The script will read the raw CSV, execute all cleaning steps, display the comparison table, and generate `healthcare-dataset-stroke-data-cleaned.csv`.

---

## Oral Recitation and Presentation Defense Guide

Be prepared to answer these core questions during the classroom presentation and oral recitation:

### Question 1: Why didn't you simply use `df.dropna()` to remove the 201 rows with missing BMI?
* **Answer:** "In this dataset, missing BMI is Not Missing Completely at Random (MCAR). The stroke incidence among patients with missing BMI is **19.90%**, compared to only **4.87%** in the overall population. In emergency clinical triage, acute stroke patients frequently cannot stand on scales to record height and weight. Dropping these 201 records would eliminate 40 stroke patients—over **16% of all positive stroke cases in the entire dataset**—inducing catastrophic selection bias."

### Question 2: Why did you recode 'Unknown' smoking status specifically for children under 10?
* **Answer:** "Exploratory analysis revealed that 100% of patients under age 10 were categorized as 'Unknown' smokers. In clinical practice, pediatric questionnaires omit tobacco usage for toddlers and young children. Treating these as missing adult survey answers distorts behavioral analysis. Recoding them to 'never smoked' correctly represents pediatric reality."

### Question 3: Why are average glucose levels above 200 mg/dL kept in the dataset instead of being removed as statistical outliers?
* **Answer:** "Standard statistical rules (like 1.5x IQR) would flag blood glucose above 200 mg/dL as outliers. However, in medical physiology, blood glucose levels between 200 and 271 mg/dL represent severe diabetic hyperglycemia. Diabetic vascular damage is one of the primary medical etiologies of stroke. Removing these patients would erase the very pathological mechanism we are investigating."

### Question 4: How does your stratified median imputation preserve data integrity?
* **Answer:** "Instead of replacing missing BMI with a single global average, we calculated the median within demographic and clinical subgroups defined by age bracket, gender, hypertension status, and stroke outcome. This ensures an elderly hypertensive stroke patient receives an imputed BMI typical of their clinical peer group, preserving authentic covariate correlations."

### Question 5: Why are some ages recorded with decimal fractions (e.g., 0.08, 0.16, 0.24)?
* **Answer:** "In pediatric healthcare datasets, ages under 1 year are recorded as fractions of a year (e.g., 0.08 years equals 1 month, 0.16 years equals 2 months). These are genuine clinical representations of infant age, not data corruption."
