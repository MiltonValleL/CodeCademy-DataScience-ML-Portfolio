# 🏠 King County Housing Price Prediction: A Multi-Model Linear Approach

<br>

## 👋 Overview
Welcome to my second Housing Price Prediction project! This repository features an end-to-end machine learning pipeline designed to estimate property values in King County, Washington (including the Seattle area), using a dataset of over 21,000 sales records.

Building upon the foundations of my previous work, this project focuses on **feature engineering**, **geographic impact analysis**, and a comparative study between **Ordinary Least Squares (OLS)** and **Regularized Models (Ridge & Lasso)** to ensure model stability and prevent overfitting.

<br>
<br>

## 🚀 Key Highlights & Methodology
This project implements a "Production-Ready" mindset, focusing on how different linear architectures respond to high-dimensional data:

---

### 1. Forensic Exploratory Data Analysis (EDA)
- **Temporal Dynamics**: Beyond simple cleaning, I extracted the `year` and `month` from sales dates to analyze market seasonality, identifying how timing affects transaction volume and pricing.
- **Geospatial Insights**: Analyzed the distribution of prices across coordinates, acknowledging that in real estate, location (Latitude/Longitude) acts as a non-linear proxy for socio-economic variables.
- **Extreme Value Management**: Identified and handled outliers in square footage and price to ensure the regression hyperplane wasn't biased by "trophy homes" or distressed sales.

---

### 2. Strategic Data Preparation
- **Feature Engineering**: Transformed the raw data into predictive signals, such as calculating the age of the house at the time of sale and identifying if a renovation had occurred.
- **Feature Selection & Pruning**: Removed non-informative identifiers (like `id`) and redundant features to reduce the "Curse of Dimensionality."
- **Data Pipeline**: Implemented a robust scaling process using `StandardScaler`, crucial for the fair evaluation of Lasso and Ridge coefficients.

---

### 3. Model Training & Evaluation (The Triple-Benchmark)
I implemented and compared three distinct architectures to find the optimal balance between bias and variance:
- **Linear Regression (Baseline)**: Established our performance floor and verified basic Gauss-Markov assumptions.
- **Ridge Regression (L2)**: Used to mitigate multicollinearity between highly correlated features like `sqft_living` and `grade`.
- **Lasso Regression (L1)**: Applied for automated feature selection, effectively "zeroing out" noise-heavy variables to enhance model parsimony.

---

<br>
<br>

## 📊 Results & Insights

---

### Model Performance
The models showed consistent performance, proving the robustness of the feature set:

- **R² Score (Test Set)**: **`~ 0.70`**
- **MAE (Mean Absolute Error)**: Approximately **`$ 124,000`**

---

### Key Drivers of Price
Based on the standardized coefficients, the primary price determinants in King County are:

- **Square Footage (sqft_living)**: The most significant predictor. Size remains the primary driver of value in the region.
- **Grade & Condition**: The construction quality assigned by the county system proved more predictive than the actual age of the house.
- **The Waterfront Premium**: A binary feature that, despite its low frequency, exerts a massive localized "multiplier effect" on valuation.

---

<br>
<br>

## 🛠️ Technologies Used
- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Environment**: Jupyter Notebook / Ubuntu 24.04 LTS

---

<br>
<br>

## 📂 Repository Structure

---

```text
├── datasets/
│   └── kc_house_data.csv      (Original Dataset)
├── Notebooks/
│   └── Housing_Prices_Project(Milton-Valle).ipynb      (Main Analysis, Regularization & Modeling)
├── README.md      (Project documentation)
└── requirements.txt      (Dependencies)
```
<br>
<br>

## 🔮 Strategic Recommendation (Business Perspective)
To provide Senior-level value, the project concludes with a deployment strategy based on specific use cases:

  - For High-Volume Operations: Lasso Regression is recommended. It maintains high accuracy while simplifying the feature set, reducing data collection costs and complexity for future predictions.

  - For Luxury Market Analysis: Linear Regression (OLS) showed slightly better resilience when handling high-end outliers, making it safer for premium valuations where underpricing carries a high opportunity cost.

<br>
<br>

## 🤝 Contact
I am a Data Science student on a mission to become a world-class professional by 2026. My work focuses on the intersection of statistical rigor and business impact.

If you have any questions about this project or would like to discuss Machine Learning, feel free to reach out!

**Author:** Milton Rodolfo Valle Lora

**LinkedIn:** [Please click here](https://www.linkedin.com/in/miltonvallelora/)
