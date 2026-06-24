# Laptop Price Predictor

A machine learning model that predicts laptop prices based on hardware specifications like RAM, CPU, GPU, storage, and display features, engineered from raw text columns into structured numeric features.

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://laptop-price-predictor-hbmp7kv66feoljdqsvukyz.streamlit.app/)

## Tech Stack

- **Python** — core language
- **Pandas & NumPy** — data cleaning and feature engineering
- **Scikit-learn** — model training and evaluation (Random Forest, Gradient Boosting, Linear Regression)
- **Seaborn & Matplotlib** — exploratory data analysis
- **Streamlit** — live web app deployment

## Dataset

The dataset (`laptop_data.csv`) contains ~1300 laptops with the following raw columns:

`Company`, `TypeName`, `Inches`, `ScreenResolution`, `Cpu`, `Ram`, `Memory`, `Gpu`, `OpSys`, `Weight`, `Price`

## Feature Engineering

Since most columns are raw strings, significant preprocessing was applied to extract usable features:

| Raw Column | Extracted Features |
|---|---|
| `ScreenResolution` | `ppi`, `touchscreen`, `IPS Panel` |
| `Cpu` | `cpu brand` (i3/i5/i7/AMD/other) |
| `Memory` | `layer1HDD`, `layer1SSD`, `layer2HDD`, `layer2SSD`, `layer1Hybrid`, `layer1FlashStorage` |
| `Ram` | Stripped `GB` → integer |
| `Weight` | Stripped `kg` → float |

## Model Performance

| Model | R² Score | MAE |
|---|---|---|
| Linear Regression | 0.79 | ₹18,659 |
| Random Forest | 0.89 | ₹12,123 |
| Gradient Boosting | 0.90 | ₹11,477 |

Gradient Boosting gave the best results with an R² of 0.90 and a mean absolute error of ₹11,477.

## Top Predictive Features

Based on Random Forest feature importances:

1. RAM
2. Weight
3. Display PPI
4. Laptop type
5. CPU clock speed
6. SSD size

## Project Structure

```
├── laptop_data.csv
├── Untitled.ipynb                  # EDA and feature engineering notebook
├── laptop_price_prediction.py      # Clean training pipeline
├── app.py                          # Streamlit app
└── README.md
```

## How to Run Locally

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib streamlit
   ```

2. Place `laptop_data.csv` in the project directory.

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

   Or run just the training script:
   ```bash
   python laptop_price_prediction.py
   ```

## Known Issues Fixed

- `sns.distplot()` is deprecated — replaced with `sns.histplot(..., kde=True)`
- `fetch_processor()` had a broken `or "string"` condition — fixed using `in [...]`
- `str.replace(r"\D", " ")` needed `regex=True` and `.str.strip()` before casting to int
