from flask import Flask, render_template
import numpy as np
import pandas as pd
from flask import request
import matplotlib.pyplot as plt
from utility import dataframe_describe,list_stock_files,plot_function
app = Flask(__name__)
stock_folder_path = "./stock_datas"



@app.route('/')
def landing_page():
    stocks_files = list_stock_files(stock_folder_path)
    return render_template(
        'index.html',
        stock_length=len(stocks_files),
        stocks=stocks_files
        )


@app.route('/forecasting', methods=['POST'])
def forecasting():
    selected_stock = request.form.get('selected_stock')
    ml_algorithm = request.form.get('selected_algorithm')
    described_data = dataframe_describe(selected_stock)
    plot_base64 = plot_function(selected_stock)
    # Call your forecasting function with the selected stock
    # You can pass the selected stock as an argument to your forecasting function here
    # Example: forecast_result = your_forecasting_function(selected_stock)
    return render_template("stock_info.html",
                           stock_name = selected_stock.upper(),
                           described_data=described_data,
                           plot_base64=plot_base64,
                           ml_algorithm=ml_algorithm
                           )

@app.route('/about',methods=['GET'])
def about_team():
    return render_template("team.html")

if __name__ == '__main__':
    app.run(debug=True)
