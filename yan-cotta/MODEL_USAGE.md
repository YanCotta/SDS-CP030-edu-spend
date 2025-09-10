# TCA Predictor Model Usage Guide

## Loading the Model
```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('tca_predictor.joblib')
```

## Making Predictions
```python
# Prepare input data as DataFrame with all required columns
input_data = pd.DataFrame({
    'Country': ['USA'],
    'City': ['New York'],
    'Program': ['Computer Science'],
    'Level': ['Masters'],
    'Duration_Years': [2.0],
    'Living_Cost_Index': [120],
    'Exchange_Rate': [1.0],
    'Tuition_USD': [50000],
    'Rent_USD': [2500],
    'Visa_Fee_USD': [500],
    'Insurance_USD': [2000]
})

# Make prediction
predicted_tca = model.predict(input_data)
print(f"Predicted TCA: ${predicted_tca[0]:,.0f}")
```

## Required Features
The model expects these features:
- Categorical: Country, City, Program, Level
- Numerical: Duration_Years, Living_Cost_Index, Exchange_Rate, Tuition_USD, Rent_USD, Visa_Fee_USD, Insurance_USD

## Model Performance
- Cross-validation R²: 0.9945 ± 0.0037
- Cross-validation MAE: $833
- Model Type: RandomForestRegressor with preprocessing pipeline

## Notes
- The model handles missing values automatically
- Cities not in the top 30 are grouped as 'Other_City'
- All preprocessing is included in the pipeline
