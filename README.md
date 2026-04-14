# Customer Churn Prediction Project

A customer churn prediction pipeline built for retention dataset.

## Project Structure

```
DS_Project_2/
├── config.py              # Configuration constants (paths, mappings, parameters)
├── main.py                # Main pipeline script (runs end-to-end)
├── requirements.txt       # Python dependencies
├── src/
│   └── util.py            # Simple utility/helper functions
├── data/
│   ├── raw/               # Raw data files (BoB.xlsx, Retention.csv)
│   └── processed/         # Cleaned and engineered data
├── models/                # Saved model .pkl files
├── notebooks/
│   ├── 00_data_cleaning.ipynb
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_building.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_model_explainability.ipynb
├── output/                # Output CSV files and plots
└── reports/               # Reports (placeholder)
```

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Place raw data files in `data/raw/`:
   - `Retention.csv`
   - `BoB.xlsx`

3. Run notebooks in order (00 through 06), OR run the full pipeline:
   ```
   python main.py
   ```


## Models Used

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

## Key Features

- statistical hypotheses tested with Mann-Whitney U and Chi-Square tests
- Feature engineering: time features, categorical encoding, derived ratios, BoB merge
- Feature selection: Random Forest importance + multicollinearity removal
- Model explainability: SHAP values
- Business output: priority banding (P1-P4), customer risk table, retention playbook
