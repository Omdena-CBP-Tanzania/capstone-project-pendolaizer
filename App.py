


import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load your cleaned climate dataset (if needed)
data_path = 'data/cleaned/cleaned_climate_data.csv'
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    st.warning('Data file not found. Please ensure it is in the right location.')
    df = pd.DataFrame()

# Load your trained model
model_path = 'models/climate_rf_model.pkl'
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.warning('Model file not found. Please ensure it is in the right location.')
    model = None

st.title('Climate Temperature Prediction App ')
st.write('This app visualizes climate data and predicts future temperatures for Tanzania.')

if not df.empty:
    st.header('Exploratory Data Analysis ')

    # Display basic data info
    st.write('### Raw Climate Data')
    st.dataframe(df.head())

    # Plot historical temperature trends if available
    if 'Year' in df.columns and 'temperature' in df.columns:
        st.write('### Temperature Trend Over the Years')
        fig, ax = plt.subplots()
        ax.plot(df['Year'], df['temperature'], marker='o', linestyle='-', color='skyblue')
        ax.set_xlabel('Year')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Trend')
        st.pyplot(fig)
    else:
        st.warning("No 'Year' or 'temperature' column found for plotting.")

st.header('Predict Future Temperature ')

# Input fields for prediction
year = st.number_input('Enter Year', min_value=2025, max_value=2100, value=2025)
month = st.number_input('Enter Month', min_value=1, max_value=12, value=1)

if st.button('Predict Temperature'):
    if model is not None:
        # Create DataFrame for prediction
        input_data = pd.DataFrame({
            'year': [year],
            'month': [month]
        })

        # Make the prediction
        predicted_temp = model.predict(input_data)[0]
        st.success(f'Predicted Temperature for {month}/{year}: {predicted_temp:.2f} °C')

        # Optional: plot future prediction (single point)
        fig, ax = plt.subplots()
        ax.bar([f'{month}/{year}'], [predicted_temp], color='orange')
        ax.set_ylabel('Predicted Temperature (°C)')
        ax.set_title('Predicted Temperature')
        st.pyplot(fig)
    else:
        st.warning('No model loaded! Please ensure the model file exists.')

#st.write("---")
#st.write("Made with  by [Your Name] | [Link to your GitHub Repo]")

