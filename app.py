from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    # Load the DataFrame from a pickle file (replace 'your_df.pkl' with the actual file path)
    df = pd.read_pickle('df.pkl')

    # Extract unique values for dropdown options
    company_options = df['Company'].unique()
    type_options = df['TypeName'].unique()
    ram_options = [2,4,6,8,12,16,24,32,64]
    cpu_options = df['CPU Brand'].unique()
    gpu_options = df['GPU Brand'].unique()
    os_options = df['OS'].unique()


    # Render an HTML template with the dropdowns
    return render_template('index.html', company_options=company_options, type_options=type_options,
                           ram_options = ram_options,
                           cpu_options = cpu_options,
                           gpu_options = gpu_options,
                           os_options = os_options)

@app.route('/predict', methods = ["POST"])

def predict():
    company = request.form.get("company")
    type = request.form.get("type")
    ram = request.form.get("ram")
    weight = request.form.get("weight")
    touchscreen = request.form.get("touchscreen")
    ips = request.form.get("ips")
    screen_size = float(request.form.get("screen_size"))
    resolution = request.form.get("resolution")
    cpu = request.form.get("cpu")
    hdd = request.form.get("hdd")
    ssd = request.form.get("ssd")
    gpu = request.form.get("gpu")
    os = request.form.get("os")

    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)

    pipe = pickle.load(open("pipe.pkl", 'rb'))
    result = int(np.exp(pipe.predict(query)))
    result = "Hi There!! The expected laptop price based on the specifications provided is Rs." + str(result)
    df = pd.read_pickle('df.pkl')

    # Extract unique values for dropdown options
    company_options = df['Company'].unique()
    type_options = df['TypeName'].unique()
    ram_options = [2, 4, 6, 8, 12, 16, 24, 32, 64]
    cpu_options = df['CPU Brand'].unique()
    gpu_options = df['GPU Brand'].unique()
    os_options = df['OS'].unique()

    return render_template("index.html",company_options=company_options, type_options=type_options,
                           ram_options = ram_options,
                           cpu_options = cpu_options,
                           gpu_options = gpu_options,
                           os_options = os_options, result = result)


if __name__ == '__main__':
    app.run()
