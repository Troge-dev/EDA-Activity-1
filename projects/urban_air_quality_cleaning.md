# Laboratory Activity 1: Data Cleaning - Urban Air Quality Assessment

## Project Overview

This repository contains the collaborative data cleaning project for **Laboratory Activity 1: Data Cleaning**, submitted under the **Environment** domain. 

Continuous ambient air quality monitoring stations (CAAQMS) deployed across major metropolitan areas record atmospheric concentrations of critical pollutants such as fine particulate matter (PM2.5), coarse particulate matter (PM10), Nitrogen Dioxide (NO2), and Carbon Monoxide (CO). However, raw environmental telemetry frequently suffers from transmission dropouts, instrument calibration drift, sensor outages, and recording inconsistencies. 

This project demonstrates a systematic, domain-informed data cleaning workflow using Python and Pandas on daily urban air quality records.

---

## Group Information and Member Roles

| Group Member | Student ID | Primary Responsibility |
| :--- | :--- | :--- |
| Member 1 (Lead) | `[Student ID]` | Data Audit, Datatype Casting, Pipeline Architecture |
| Member 2 | `[Student ID]` | Outlier Investigation, Physical Domain Boundary Checks |
| Member 3 | `[Student ID]` | Missing Value Imputation Strategy, Time-Series Logic |
| Member 4 | `[Student ID]` | Categorical Standardization, Before vs. After Reporting |

---

## Research Question and Problem Statement

### The Real-World Environmental Problem
Urban air pollution is a major environmental health crisis, linked directly to cardiovascular and respiratory illnesses. While automated monitoring networks provide open data, raw measurements cannot be directly used for public health policy or predictive modeling without rigorous data quality audits.

### Core Investigation Question
> *How reliable are daily urban air quality records across major metropolitan areas from 2015 to 2020, and how can we methodically identify and clean sensor errors, negative concentrations, and missing telemetry while preserving genuine hazardous pollution events such as winter inversions and agricultural stubble burning?*

---

## Dataset Profile

* **Dataset Title (Kaggle):** Air Quality Data in India (2015 - 2020)
* **Dataset Creator:** Rohit Sahoo (Source: Central Pollution Control Board - CPCB)
* **Primary File:** `city_day.csv` (Located in the root directory)
* **Dimensions:** 29,531 rows × 16 columns
* **Time Span:** January 1, 2015 to July 1, 2020

### Key Variables Analyzed

| Variable | Raw Data Type | Unit / Scale | Environmental Significance |
| :--- | :--- | :--- | :--- |
| `City` | `object` | Categorical | Geographic metropolitan monitoring area |
| `Date` | `object` | `YYYY-MM-DD` | Daily timeline of continuous observation |
| `PM2.5` | `float64` | ug/m3 | Fine inhalable particles (<= 2.5 um); primary health hazard |
| `PM10` | `float64` | ug/m3 | Coarse inhalable particles (<= 10 um); dust and road debris |
| `NO2` | `float64` | ug/m3 | Nitrogen Dioxide; marker of vehicular and industrial emissions |
| `CO` | `float64` | mg/m3 | Carbon Monoxide; marker of incomplete combustion |
| `AQI` | `float64` | Index (0 - 500+) | Composite National Air Quality Index |
| `AQI_Bucket` | `object` | Categorical | Health hazard category (Good to Severe) |

---

## Identified Data Quality Problems

During initial exploratory auditing via `df.info()`, `df.describe()`, and `df.isna().sum()`, five main data quality issues were identified:

1. **Incorrect Datatypes:**
   * `Date` is stored as an `object` (string) rather than a temporal datatype (`datetime64[ns]`), preventing chronological sorting and time-window indexing.
2. **Text and Categorical Inconsistencies:**
   * `City` names and `AQI_Bucket` categories contain inconsistent casing and trailing whitespace.
3. **Physically Impossible Sensor Values:**
   * Baseline drift and voltage irregularities in electrochemical and optical sensors produced negative concentration values (e.g., negative PM2.5 and NO2 readings).
4. **Non-Random Missing Values:**
   * Periodic power cuts, sensor calibration downtime, and station maintenance caused missing data rates of 15% to 35% across pollutant features.
5. **Extreme Positive Outliers:**
   * Particulate matter readings reach extreme peaks (> 600 ug/m3 in winter months), requiring domain verification to avoid accidentally discarding real environmental disasters.

---

## Data Cleaning Methodology and Domain Rationale

The cleaning pipeline is structured into five distinct phases, documented below with reproducible Python code and scientific justifications.

### Step 1: Datatype Conversion and Temporal Sorting

* **Action:** Convert `Date` to `pd.to_datetime` and sort the dataset chronologically per city.
* **Code:**
```python
import pandas as pd
import numpy as np

df = pd.read_csv("../city_day.csv")

# Convert string dates to datetime objects
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

# Sort chronologically by city and date
df = df.sort_values(by=["City", "Date"]).reset_index(drop=True)
```
* **Rationale:** Time-series operations such as rolling averages and localized forward-filling require strict chronological sequencing.

---

### Step 2: Categorical and Text Standardization

* **Action:** Strip leading/trailing whitespaces and enforce uniform title casing.
* **Code:**
```python
# Clean text columns
df["City"] = df["City"].str.strip().str.title()
df["AQI_Bucket"] = df["AQI_Bucket"].str.strip().str.title()
```
* **Rationale:** Eliminates duplicate grouping categories caused by trailing spaces or case differences (e.g., `"Delhi "` vs `"Delhi"`).

---

### Step 3: Handling Physically Impossible Values (Negative Concentrations)

* **Action:** Identify all negative pollutant concentrations and replace them with `np.nan`.
* **Code:**
```python
pollutant_columns = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]

for col in pollutant_columns:
    # Mass concentration cannot be below zero
    invalid_mask = df[col] < 0
    df.loc[invalid_mask, col] = np.nan
```
* **Domain Rationale:** Concentration represents physical mass per unit volume (ug/m3). Negative values are hardware calibration artifacts resulting from ambient humidity shifts or zero-point sensor drift. They are converted to missing values to prevent downward bias in rolling averages.

---

### Step 4: Domain-Aware Missing Value Imputation

* **Action:** Apply bounded linear interpolation grouped by `City`, followed by localized forward-filling.
* **Code:**
```python
# Time-continuity interpolation within each city (maximum 3 consecutive days)
df[pollutant_columns] = df.groupby("City")[pollutant_columns].transform(
    lambda group: group.interpolate(method="linear", limit=3).ffill().bfill()
)
```
* **Domain Rationale:** 
  * Atmospheric pollutants exhibit high temporal autocorrelation: today's air quality is strongly correlated with yesterday's weather pattern.
  * Global mean or median imputation is inappropriate because it would blend clean monsoon air with winter smog, destroying seasonal variance.
  * Interpolation is strictly grouped by `City` to prevent data leakage between geographically separated monitoring stations.

---

### Step 5: Validating Outliers vs. Genuine Environmental Events

* **Domain Decision:** Retain verified high particulate readings (> 600 ug/m3 for PM2.5).
* **Domain Rationale:**
  * In northern cities (e.g., Delhi, Lucknow, Patna) during late October to December, post-monsoon agricultural crop residue burning combined with low boundary-layer temperature inversions traps particulates at surface level.
  * Removing these values through standard IQR or Z-score filtering would remove the most dangerous public health incidents from the dataset, producing a false picture of air safety.

---

### Step 6: Harmonizing AQI and AQI Categories

* **Action:** Recalculate missing `AQI_Bucket` labels according to standard national air quality index ranges.
* **Code:**
```python
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

df["AQI_Bucket"] = df["AQI"].apply(classify_aqi)
```
* **Rationale:** Ensures 100% categorical alignment between numeric AQI index values and risk buckets.

---

## Before vs. After Summary

| Quality Check / Metric | Raw Dataset (`city_day.csv`) | Cleaned Dataset |
| :--- | :--- | :--- |
| **`Date` Format** | `object` (string) | `datetime64[ns]` |
| **Record Sorting** | Unsorted / mixed entries | Sorted chronologically by `City` then `Date` |
| **Negative Sensor Values** | Present in multiple columns | 0 (Replaced and interpolated) |
| **Missing `PM2.5` Records** | 4,598 missing values (15.57%) | 0 missing in active station windows |
| **Missing `AQI_Bucket` Records** | 4,681 missing values (15.85%) | 0 missing (Classified from AQI) |
| **Text Standardization** | Mixed casing and trailing spaces | Uniform Title Case without whitespace |
| **Downstream Utility** | Fails time-series models; skewed by errors | Ready for seasonal decomposition and EDA |

---

## How to Run the Cleaning Pipeline

### Prerequisites
* Python 3.8+
* pandas
* numpy

Install requirements:
```bash
pip install pandas numpy
```

### Execution
Run the cleaning script from your terminal or notebook environment:
```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("../city_day.csv")

# Run full pipeline
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df = df.sort_values(by=["City", "Date"]).reset_index(drop=True)
df["City"] = df["City"].str.strip().str.title()

pollutants = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
for col in pollutants:
    df.loc[df[col] < 0, col] = np.nan

df[pollutants] = df.groupby("City")[pollutants].transform(
    lambda g: g.interpolate(method="linear", limit=3).ffill().bfill()
)

def classify_aqi(aqi):
    if pd.isna(aqi): return "Unknown"
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"

df["AQI_Bucket"] = df["AQI"].apply(classify_aqi)

# Save cleaned output
df.to_csv("city_day_cleaned.csv", index=False)
print("Data cleaning complete. Output saved to city_day_cleaned.csv.")
```

---

## Oral Recitation and Presentation Defense Guide

Be prepared to answer these core questions during the oral defense:

### Question 1: How did your group identify dirty data in the raw CSV?
* **Answer:** "We used `df.info()` to detect incorrect object datatypes on dates, `df.isna().sum()` to quantify missing sensor rates per pollutant, and `df.describe()` to check statistical minimums, which exposed impossible negative concentrations."

### Question 2: Why did you not use simple mean imputation to fill missing values?
* **Answer:** "Air pollution has strong seasonal and meteorological dependency. A city's annual average blends clean monsoon months with hazardous winter periods. Mean imputation would distort natural seasonal cycles, so we used localized time-series linear interpolation within each city."

### Question 3: How did domain knowledge stop you from deleting extreme high values?
* **Answer:** "Standard statistical outlier rules like 3x IQR would flag PM2.5 readings above 600 ug/m3 as anomalies. However, in atmospheric science, winter temperature inversions and crop burning create genuine air quality emergencies of that magnitude. Deleting them would result in severe survivorship bias."

### Question 4: Why was it necessary to group by City during interpolation?
* **Answer:** "Air quality in coastal cities like Chennai or Mumbai is governed by maritime breezes, whereas inland cities like Delhi experience continental entrapment. Grouping by city ensures that data from one geographic region never leaks into another during interpolation."
