import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the trained models
regressor = joblib.load(r'D:\demo\smart_price\smart_price_regressor.pkl')
classifier = joblib.load(r'D:\demo\smart_price\smart_price_classifier.pkl')

st.title("Smart Local Price Negotiation Advisor")
st.write("Enter the details of your product to get price predictions and negotiation advice.")

Product = st.selectbox("Product Type", ["Gadget", "Clothing", "Furniture", "Book", "Toy"])
Category = st.selectbox("Category", ["Electronics", "Clothing", "Home Appliances", "Books", "Toys"])
Condition = st.selectbox("Condition", ["New", "Like New", "Used", "Heavily Used"])
Brand = st.selectbox("Brand", ["Brand A", "Brand B", "Brand C", "Brand D"])
model = st.selectbox("Model", ["Basic", "Premium", "Luxury"])
Urgency = st.selectbox("Urgency", ["Low", "Medium", "High"])
local_demand = st.selectbox("Local Demand", ["Low", "Medium", "High"])
Region = st.selectbox("Region", ["North", "South", "East", "West"])
product_age = st.number_input("Product Age (in months)", min_value=0, max_value=120, step=1)
price = st.number_input("Expected Price", min_value=0.0, step=0.01)

if st.button("🔮 Predict Fair Price & Negotiation Feasibility"):

    # Map urgency and demand to numbers
    urgency_map = {"Low": 0, "Medium": 1, "High": 2}
    demand_map = {"Low": 0, "Medium": 1, "High": 2}

    input_data = pd.DataFrame([{
        "Product": Product,  # string
        "Brand": Brand,      # string
        "Model": model,      # string
        "Condition": Condition,  # string
        "Region": Region,        # string
        "Age": product_age,      # numeric
        "Seller_Urgency": urgency_map[Urgency],         # numeric
        "Original_Price": price,                        # numeric
        "Listings_in_Region": 200,                      # numeric
        "Local_Demand_Score": demand_map[local_demand]  # numeric
    }])

    predicted_price = regressor.predict(input_data)[0]
    st.write(f"Predicted Price: ${predicted_price:.2f}")

    negotiation_advice = classifier.predict(input_data)[0]
    st.write(f"Negotiation Advice: {'Negotiable' if negotiation_advice==1 else 'Fixed Price'}")

    # SHAP explainability (TreeExplainer for XGBoost)
    shap.initjs()
    explainer = shap.Explainer(regressor.named_steps['regressor'])  # Access XGBRegressor from pipeline
    X_transformed = regressor.named_steps['preprocessor'].transform(input_data)
    shap_values = explainer(X_transformed)

    # Display SHAP explanation in Streamlit
    st.subheader("🔍 Why this price? Feature Contribution (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)
    
    st.markdown("### 📊 Feature Importance (Bar)")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig2)


