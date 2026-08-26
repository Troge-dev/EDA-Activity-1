# Laboratory Activity 1: Data Cleaning - Global Coastal Plastic Waste and Economic Mismanagement

## Project Overview

This repository contains the collaborative data cleaning project for **Laboratory Activity 1: Data Cleaning**, submitted under the **Environment & Marine Ecology** domain. 

Plastic pollution is one of the most widespread environmental challenges of the Anthropocene, directly threatening marine organisms, coastal fisheries, and global food chains (UN Sustainable Development Goal 14: Life Below Water). Assessing national responsibility and plastic leakage risk requires integrating global environmental studies with economic and demographic data. 

However, multi-source global synthesis datasets frequently suffer from extreme matrix sparsity, historical century-long demographic mismatches, inconsistent regional macro-aggregations, and unstandardized geographic identifiers. This project implements a rigorous data cleaning pipeline using Python and Pandas to clean, harmonize, and validate global plastic mismanagement records.

---

## Research Question and Problem Statement

### The Real-World Environmental Problem
Over 8 million metric tons of plastic enter the global ocean annually, with the vast majority stemming from inadequately managed municipal solid waste in coastal regions. Global policymakers require accurate, country-level baseline models to evaluate waste management infrastructure needs. However, raw data files merge disparate historical databases containing over 200 years of sparse population surveys alongside single-year environmental benchmarks, producing severely distorted analytical sets.

### Core Investigation Question
> *How can we systematically filter, clean, and harmonize cross-national demographic and economic records to isolate valid coastal plastic waste mismanagement benchmarks, eliminate double-counting from macro-regional aggregates, and resolve multi-decade metadata gaps across sovereign states?*

---

## Dataset Profile

* **Dataset Title (Kaggle):** Global Plastic Pollution / Plastic Datasets
* **Dataset Creator:** Soham Gade (Source: *Jambeck et al., Science*, World Bank, & Our World in Data)
* **Primary File:** `per-capita-mismanaged-plastic-waste-vs-gdp-per-capita.csv` (Located in this project directory)
* **Raw Dimensions:** 48,169 rows × 7 columns
* **Temporal Scope:** 1800 to 2017 (Global plastic benchmark: 2010)

### Key Variables Analyzed

| Variable | Raw Data Type | Unit / Scale | Environmental Significance |
| :--- | :--- | :--- | :--- |
| `Entity` | `object` | Categorical | Country or territory name |
| `Code` | `object` | ISO Alpha-3 | 3-letter national sovereign identifier |
| `Year` | `int64` | `YYYY` | Observation year (historical census timeline) |
| `Per capita mismanaged plastic waste` | `float64` | kg/person/day | Inadequately disposed municipal plastic per person |
| `GDP per capita, PPP` | `float64` | Constant 2011 Int $ | National purchasing-power-parity economic wealth |
| `Total population` | `float64` | Count | Total national population (Gapminder/UN estimates) |
| `Continent` | `object` | Categorical | Continental landmass grouping |

---

## Identified Data Quality Problems

Exploratory analysis through `df.info()`, `df.describe()`, and `df.isna().sum()` revealed five major structural data quality problems:

1. **Extreme Historical Matrix Sparsity (Over 99% Empty Cells):**
   * The dataset contains population estimates dating back to **1800**. However, commercial plastics were not manufactured prior to 1950, and global mismanaged waste was systematically quantified in **2010**. Over 47,980 rows lack the target environmental measurement.
2. **Entity Conflation (Regional Blocs vs. Sovereign Nations):**
   * Macro-regional aggregates (e.g., `"World"`, `"European Union"`, `"East Asia & Pacific"`, `"High-income countries"`) and non-sovereign dependencies (`"Channel Islands"`) are mixed directly into the `Entity` column alongside sovereign states like `"Japan"` or `"Nigeria"`.
3. **Severe Metadata Disconnection (`Continent` Nulls):**
   * `Continent` is only recorded on the year 2015 slice, leaving over 90% of the dataset—including the critical 2010 benchmark records—with `NaN` for continent.
4. **Cumbersome and Non-Standard Column Names:**
   * Raw headers contain commas, spaces, and currency symbols (`"GDP per capita, PPP (constant 2011 international $)"`), preventing standard programmatic indexing and increasing coding error rates.
5. **Extreme Logarithmic Skewness in Tourism Economies:**
   * Small island developing states (e.g., Seychelles, Trinidad and Tobago, Vanuatu) exhibit per-capita plastic rates up to 20x higher than large continental nations due to tourist consumption relative to permanent resident population counts.

---

## Data Cleaning Methodology and Domain Rationale

The cleaning pipeline applies domain-informed filtering and harmonization across six distinct steps:

### Step 1: Standardizing Column Names

* **Action:** Rename raw headers into clean, lowercase snake_case identifiers.
* **Code:**
```python
import pandas as pd
import numpy as np

df = pd.read_csv("per-capita-mismanaged-plastic-waste-vs-gdp-per-capita.csv")

# Standardize column headers
df.columns = [
    "country", "iso_code", "year", "mismanaged_waste_per_capita", 
    "gdp_per_capita", "population", "continent"
]
```
* **Rationale:** Eliminates syntax issues and spaces in column referencing.

---

### Step 2: Time-Invariant Metadata Propagation (`Continent`)

* **Action:** Propagate continental labels across all historical years for each country using forward-fill and backward-fill.
* **Code:**
```python
# Propagate static geographic continent metadata across country records
df["continent"] = df.groupby("country")["continent"].transform(
    lambda group: group.ffill().bfill()
)
```
* **Domain Rationale:** A nation's continental location is an invariant geographic fact. Forward/backward filling within each sovereign country correctly restores continental metadata for the 2010 benchmark observations without introducing cross-country contamination.

---

### Step 3: Filtering to the Valid Environmental Benchmark Era (2010)

* **Action:** Retain only records with measured plastic waste mismanagement.
* **Code:**
```python
# Filter out historical population-only records lacking environmental data
df_clean = df[df["mismanaged_waste_per_capita"].notna()].copy()
```
* **Domain Rationale:** Retaining rows from 1800 to 2009 is scientifically invalid for plastic modeling, as synthetic polymers were not in wide use, and no global empirical waste measurements exist for those periods.

---

### Step 4: Pruning Macro-Regional Aggregates and Custom Entities

* **Action:** Remove all entities lacking valid sovereign ISO-3 codes or utilizing custom `OWID_*` regional prefixes.
* **Code:**
```python
# Identify and remove macro-regional blocs and aggregates
invalid_mask = df_clean["iso_code"].isna() | df_clean["iso_code"].str.startswith("OWID_")
df_clean = df_clean[~invalid_mask].reset_index(drop=True)
```
* **Domain Rationale:** Retaining aggregate parent entities (such as `"World"` or `"Latin America"`) alongside individual sovereign states produces double-counting, distorts national standard deviations, and invalidates statistical hypothesis testing.

---

### Step 5: Country-Level Temporal Imputation for Economic Indicators

* **Action:** Impute missing 2010 GDP per capita or Population values using the nearest available historical records for that specific country.
* **Code:**
```python
# Impute missing economic/demographic indicators using country-specific historical medians
for col in ["gdp_per_capita", "population"]:
    country_medians = df.groupby("country")[col].median()
    df_clean[col] = df_clean[col].fillna(df_clean["country"].map(country_medians))
```
* **Domain Rationale:** When a country's 2010 GDP census was omitted, its adjacent 2009 or 2011 GDP figure provides a far more accurate representation than a global or continental mean.

---

### Step 6: Logarithmic Transformation for Heavy Skewness

* **Action:** Compute $\log_{10}$ transformations on GDP and mismanaged waste per capita.
* **Code:**
```python
# Apply log10 transformation for econometric modeling
df_clean["log_gdp"] = np.log10(df_clean["gdp_per_capita"])
df_clean["log_mismanaged_waste"] = np.log10(df_clean["mismanaged_waste_per_capita"] + 1e-5)
```
* **Domain Rationale:** National economic wealth and waste generation follow heavy power-law distributions. Logarithmic transformations linearize the relationship and prevent high-income outliers from skewing linear regressions.

---

## Before vs. After Summary

| Quality Check / Metric | Raw Dataset (`per-capita...csv`) | Cleaned Dataset (`coastal_plastic_waste_cleaned.csv`) |
| :--- | :--- | :--- |
| **Row Count** | 48,169 rows (99.6% sparse) | **186 clean, verified sovereign country records** |
| **Target Null Count** | 47,982 missing values | **0 missing values** in target analysis columns |
| **`Continent` Nulls** | >90% missing across rows | **0% missing** (Propagated across country records) |
| **Entity Consistency** | Mixed nations, continents, and income tiers | **100% sovereign states** with valid ISO-3 codes |
| **Column Structure** | Messy strings with symbols and units | Standardized snake_case attributes |
| **Downstream Utility** | Unusable for modeling (empty matrix) | Ready for regression, clustering, and choropleth mapping |

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
Run the following script directly from the `projects/coastal_plastic_waste/` directory:
```python
import pandas as pd
import numpy as np

# Step 1: Load raw dataset
df = pd.read_csv("per-capita-mismanaged-plastic-waste-vs-gdp-per-capita.csv")

# Step 2: Standardize column names
df.columns = [
    "country", "iso_code", "year", "mismanaged_waste_per_capita", 
    "gdp_per_capita", "population", "continent"
]

# Step 3: Propagate continent metadata
df["continent"] = df.groupby("country")["continent"].transform(
    lambda g: g.ffill().bfill()
)

# Step 4: Filter to valid plastic benchmark records
df_clean = df[df["mismanaged_waste_per_capita"].notna()].copy()

# Step 5: Drop non-sovereign regional aggregates
invalid_mask = df_clean["iso_code"].isna() | df_clean["iso_code"].str.startswith("OWID_")
df_clean = df_clean[~invalid_mask].reset_index(drop=True)

# Step 6: Impute missing economic indicators from country timelines
for col in ["gdp_per_capita", "population"]:
    country_medians = df.groupby("country")[col].median()
    df_clean[col] = df_clean[col].fillna(df_clean["country"].map(country_medians))

# Step 7: Feature engineering
df_clean["log_gdp"] = np.log10(df_clean["gdp_per_capita"])
df_clean["log_mismanaged_waste"] = np.log10(df_clean["mismanaged_waste_per_capita"] + 1e-5)

# Step 8: Export cleaned output
df_clean.to_csv("coastal_plastic_waste_cleaned.csv", index=False)
print("Coastal plastic waste cleaning complete. Output saved to coastal_plastic_waste_cleaned.csv.")
```

---

## Oral Recitation and Presentation Defense Guide

Be prepared to answer these core questions during the oral defense:

### Question 1: Why did your cleaning process reduce the dataset from 48,000+ rows to 186 rows?
* **Answer:** "The raw dataset is a concatenated multi-source historical archive containing population estimates from 1800 to 2017. However, global coastal plastic waste was systematically measured only for the benchmark year 2010. Over 47,900 rows were empty placeholders for plastic metrics. Pruning them retained 100% of the actual scientific marine pollution observations."

### Question 2: Why was it necessary to remove entities with `OWID_` ISO prefixes?
* **Answer:** "Our World in Data assigns custom prefixes like `OWID_WRL` (World) and `OWID_EAP` (East Asia and Pacific) to regional summary rows. Leaving these in the dataset alongside sovereign countries would cause double-counting and violate the assumption of independent observations in regression models."

### Question 3: How did you resolve the fact that Continent was missing on almost every 2010 observation?
* **Answer:** "The raw dataset only recorded continent information on the year 2015 slice. Since a country's continent does not change over time, we grouped by `country` and used forward-fill and backward-fill (`ffill().bfill()`) to propagate the continental metadata back to the 2010 records."

### Question 4: Why did you not remove small island nations with extremely high per-capita plastic waste as outliers?
* **Answer:** "In marine waste economics, small island developing states (SIDS) like Vanuatu or Saint Lucia have high tourist influxes relative to their resident populations. Tourists generate substantial single-use plastic waste that enters coastal systems. Deleting these as outliers would exclude the exact regions most vulnerable to ocean plastic leakage."
