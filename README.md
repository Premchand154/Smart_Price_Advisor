### Smart Local Price Negotiation Advisor 
A smart Streamlit web app that predicts a fair price for used/local products and provides negotiation advice, built using XGBoost and SHAP explainability.

---------------
### Problem Statement:
In the second-hand market, both buyers and sellers often face a common issue — what's the right price? Overpricing leads to no buyers; underpricing means losses.
This app acts like a local price advisor by analyzing key factors like brand, condition, urgency, demand, and more, helping:
1. Sellers list their product at the best price
2. Buyers understand if negotiation is even possible
3. Both parties get insights into what influenced the prediction
---
## Features:
- Predicts fair price using a trained XGBoost model  
- Suggests whether the price is negotiable or fixed  
- Shows SHAP-based feature contributions for transparency  
- Built with an interactive Streamlit interface
---
## Why XGBoost?
We tried various models (Linear Regression, Decision Tree, Random Forest), but XGBoost delivered the best performance:
- Higher accuracy with early stopping and regularization
- Efficient on tabular structured data
- Handles feature interactions better
- Natively integrates with SHAP for explainability

-----
## Tech Stack:
- Python
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Pandas, NumPy

## Setup Instructions
### 1.Clone this repo:
```bash
git clone https://github.com/yourusername/smart-price-advisor.git
cd smart-price-advisor
```
### 2.Install dependencies:
```bash
pip install -r requirements.txt
```
### 3.Run the app:
```bash
streamlit run app.py
```

## License
This project is open-source and available under the MIT License.
