from flask import Flask, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import dataframe_describe
app = Flask(__name__)

@app.route('/')
def hello_world():
    described_data = dataframe_describe()
    return render_template('index.html',df_describe_html=described_data)

if __name__ == '__main__':
    app.run(debug=True)
