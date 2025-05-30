
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load data and model
df = pd.read_csv('data/cleaned/cleaned_climate_data.csv')
model = joblib.load('models/climate_rf_model.pkl')

# Title and introduction
st.title("Tanzania Climate Analysis and Prediction")
st.markdown("""
This app provides insights into Tanzania's climate data, including temperature and rainfall trends, and uses a machine learning model to predict average temperatures.
""")

if not df.empty:
    st.header('Exploratory Data Analysis 📊')

    # Display basic data info
    st.write('### Raw Climate Data')
    st.dataframe(df.head())


# Visualizations
st.subheader("Temperature Trends")
fig, ax = plt.subplots()
ax.plot(df['Year'], df['Average_Temperature_C'], label='Average Temperature (°C)')
ax.set_xlabel('Year')
ax.set_ylabel('Average Temperature (°C)')
ax.legend()
st.pyplot(fig)

st.subheader("Rainfall Trends")
fig, ax = plt.subplots()
ax.plot(df['Year'], df['Total_Rainfall_mm'], label='Total Rainfall (mm)', color='green')
ax.set_xlabel('Year')
ax.set_ylabel('Total Rainfall (mm)')
ax.legend()
st.pyplot(fig)

# Prediction section
st.subheader("Predict Average Temperature")
year = st.slider("Select Year", int(df['Year'].min()), int(df['Year'].max()), step=1)
month = st.selectbox("Select Month", list(range(1, 13)))
if st.button("Predict"):
    prediction = model.predict([[year, month]])[0]
    st.success(f"Predicted Average Temperature for {year}-{month}: {prediction:.2f} °C")


# Input fields for prediction future Temperature
st.subheader("Predict Future Average Temperature")
year = st.number_input('Enter Year', min_value=2025, max_value=2100, value=2025)
month = st.number_input('Enter Month', min_value=1, max_value=12, value=1)

if st.button('Predict Temperature'):
    if model is not None:
        # Create DataFrame for prediction
        input_data = pd.DataFrame({
            'Year': [year],
            'Month': [month]
        })

        # Make the prediction
        predicted_temp = model.predict(input_data)[0]
        st.success(f'Predicted Temperature for {month}/{year}: {predicted_temp:.2f} °C')

# Project Deliverables
st.subheader("📦 Project Deliverables")
st.markdown("""
✅ **Final Report**: A comprehensive document summarizing project goals, methodology, EDA findings, and model performance.  
✅ **Python Code**: Scripts for data cleaning, EDA, model development, and app deployment.  
✅ **Streamlit Web App**: This live app showcasing climate trends and predictions.  
✅ **Conclusion**: The project has equipped us with skills in data preprocessing, EDA, ML modeling, and app deployment, using freely available climate data to uncover patterns in Tanzania's climate.
""")

# Conclusion or next steps
st.subheader("🔍 Next Steps / Future Work")
st.markdown("""
- Incorporate more detailed datasets (e.g., humidity, wind speed).  
- Explore seasonal variations and long-term climate change impacts.  
- Extend to other regions or countries for broader climate analysis.
""")
