# Laboratory Activity 1: Data Cleaning

## Objective

In this activity, you will apply the data-cleaning techniques discussed in Week 2 to a real-world dataset. You will investigate the dataset, identify data-quality problems, clean the data, and explain the reasoning behind your decisions.

Groupings have already been posted. Work only with your assigned group.

---

## Instructions

### I. Choose a Theme or Domain

Choose a domain that interests your group, such as:

* Agriculture
* Entertainment
* Healthcare
* Sports
* Education
* Finance
* Or any other relevant domain (e.g., Environment)

### II. Identify a Problem or Question

Within your chosen domain, identify a real-world problem or question that could be investigated using data.

**For example:**
* *Agriculture:* What factors are associated with higher crop yields?
* *Entertainment:* What factors are associated with highly rated movies?

Your problem or question should give you a clear reason for choosing your dataset.

### III. Find a Dataset

Find a dataset that can help address your chosen problem or question.

Your group should be able to explain:
* What the dataset is about
* Where it came from
* What the important variables represent
* Why you chose it
* How it relates to your problem or question

### IV. Investigate the Dataset

Before cleaning, examine the original dataset and identify its problems.

Look for issues such as:
* Missing values
* Duplicate records
* Spelling errors
* Incorrect data types
* Inconsistent values
* Invalid values
* Unusual or extreme observations

Use appropriate Python/pandas commands to investigate these problems.

### V. Clean the Dataset

Apply the appropriate data-cleaning techniques to the problems you discovered.

For every major cleaning decision, be prepared to explain:
* **What is the problem?**
* **How did you identify it?**
* **What did you do?**
* **Why did you choose that method?**

Your code should be properly organized and commented so that the cleaning process is easy to understand.

### VI. Consider Domain Knowledge

Your cleaning decisions should consider the real-world meaning of the data.

Do not automatically remove values just because they look unusual.

**Ask:**
> *Does this value actually make sense within the domain?*

Explain how your understanding of the chosen domain influenced at least some of your cleaning decisions.

### VII. Show Before and After

Present evidence of your cleaning process by showing the dataset before and after cleaning.

Explain what changed and why.

---

## Class Presentation and Oral Recitation

* Each group will present their work in class.
* Your presentation should cover everything listed above.
* After the presentation, there will be an oral recitation.
* Every member must understand the entire project. Any member may be asked to explain the dataset, code, cleaning method, or reasoning behind the group's decisions.

---

## Projects in This Repository

This repository hosts two self-contained environmental data cleaning projects, each equipped with its raw dataset, cleaning pipeline, domain-specific justifications, and oral defense guides.

### 1. Urban Air Quality Assessment

* **Directory:** [projects/urban_air_quality/](projects/urban_air_quality/README.md)
* **Dataset:** `city_day.csv` (29,531 rows × 16 columns)
* **Domain:** Atmospheric Science & Environmental Health
* **Brief Description:** This project analyzes multi-year ambient air quality telemetry across Indian metropolitan areas from the Central Pollution Control Board (CPCB). It tackles sensor calibration drift (negative concentration values), time-series continuity breaks, missing particulate readings (PM2.5, PM10, NO2), and categorical text formatting. It establishes domain rules to distinguish true severe pollution emergencies (e.g., winter atmospheric inversions and post-monsoon crop-stubble burning) from electronic sensor glitches.

### 2. Water Potability and Chemical Safety Assessment

* **Directory:** [projects/water_potability/](projects/water_potability/README.md)
* **Dataset:** `water_potability.csv` (3,276 rows × 10 columns)
* **Domain:** Aquatic Chemistry & Public Health
* **Brief Description:** This project evaluates laboratory water quality metrics to classify water drinkability according to World Health Organization (WHO) standards. It addresses selective laboratory test omission (with over 23% missing Sulfates and 15% missing pH values), probe calibration boundaries (enforcing the strict physical 0–14 pH scale), and extreme mineral solids skewness. The cleaning methodology uses class-conditional median imputation to preserve the distinct geochemical fingerprints of potable versus contaminated water bodies.
