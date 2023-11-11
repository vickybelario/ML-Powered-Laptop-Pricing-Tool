import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the DataFrame from a pickle file (replace 'your_df.pkl' with the actual file path)
df = pd.read_pickle('df.pkl')

# Extract unique values for dropdown options
company_options = df['Company'].unique()
type_options = df['TypeName'].unique()
ram_options = [2, 4, 6, 8, 12, 16, 24, 32, 64]
cpu_options = df['CPU Brand'].unique()
gpu_options = df['GPU Brand'].unique()
os_options = df['OS'].unique()
resolution_options = ["1920x1080", "1366x768", "1600x900", "3840x2160", "3200x1800", "2880x1800", "2560x1600", "2560x1440", "2304x1440"]

st.title("Laptop Price Predictor")

st.write("<h5>In this project, I engineered a Laptop Price Predictor, leveraging a comprehensive dataset of 1303 laptop entries. The project involved intricate data manipulation and feature engineering. I employed essential libraries such as Pandas, NumPy, and Scikit-Learn for data preprocessing, cleaning, and machine learning.

Data visualization was a pivotal step, and I harnessed Matplotlib and Seaborn to create insightful barplots and apply data transformations like log transformations for further analysis.

To encode categorical variables, I employed a combination of One-Hot Encoding and SimpleImputer, facilitated by Scikit-Learn's OneHotEncoder and SimpleImputer. Further, I utilized Scikit-Learn's ColumnTransformer to efficiently transform and preprocess the data, allowing me to convert non-numeric data into a format suitable for machine learning models.

For predictive modeling, I structured the process with Scikit-Learn's Pipelines, ensuring seamless and efficient data processing. I also applied Scikit-Learn's ensemble techniques, including stacking and voting classifiers, to boost prediction accuracy. What sets this project apart is the use of Pickle for model serialization, enabling the model to be saved and loaded effortlessly.

The final model was deployed as a user-friendly web app using Flask, enabling users to estimate laptop prices based on various specifications.ython (Programming Language)</h5>", unsafe_allow_html=True)

st.write("<h5>Select your desired configurations:</h5>", unsafe_allow_html=True)

# Organize input elements in a single line
col1, col2, col3, col4 = st.columns(4)

with col1:
    company = st.selectbox("Brand:", company_options)
    ram = st.selectbox("Ram (in GB):", ram_options)

with col2:
    type = st.selectbox("Type:", type_options)
    weight = st.number_input("Weight (Kgs)", step=0.01)

with col3:
    touchscreen = st.selectbox("Touchscreen:", ["No", "Yes"])
    ips = st.selectbox("IPS:", ["No", "Yes"])

with col4:
    screen_size = st.number_input("Screen Size (Inches)", step=0.01)
    resolution = st.selectbox("Resolution:", resolution_options)

# Organize remaining input elements in a single line
col5, col6, col7, col8 = st.columns(4)

with col5:
    cpu = st.selectbox("CPU:", cpu_options)
    hdd = st.selectbox("HDD (in GB):", [0, 128, 256, 512, 1024, 2048])

with col6:
    ssd = st.selectbox("SSD (in GB):", [0, 8, 128, 256, 512, 1024])
    gpu = st.selectbox("GPU:", gpu_options)

with col7:
    os = st.selectbox("OS:", os_options)

# Add the "Submit" button
if st.button("Submit"):
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    if resolution:  # Check if resolution is not empty
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    else:
        X_res = 0  # Set a default value
        Y_res = 0
        ppi = 0

    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)

    # Load the trained model (replace 'pipe.pkl' with the actual file path)
    pipe = pickle.load(open("pipe.pkl", 'rb'))

    result = int(np.exp(pipe.predict(query)))
    result_str = f"Hi There!! The expected laptop price based on the specifications provided is Rs. {result}"
    st.write(result_str, unsafe_allow_html=True)
