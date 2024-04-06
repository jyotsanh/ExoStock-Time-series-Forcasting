from flask import Flask, render_template
import numpy as np
import pandas as pd
from flask import request
import matplotlib.pyplot as plt
from utility import dataframe_describe,list_stock_files,plot_function,train_model
app = Flask(__name__)
stock_folder_path = "./STOCKS"



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
    selected_stock = request.form.get('selected_stock') #->user selected stock
    ml_algorithm = request.form.get('selected_algorithm') #->ml algorithm selected by use
    
    described_data = dataframe_describe(selected_stock)#->data.describe()
    rmse,plotted = train_model(selected_stock,ml_algorithm)
    columns = ['open','high','low']#->column for visualization
    plot_list = []
    for col in columns:#->iterate overe each columns
        
        plot_list.append(plot_function(selected_stock,col,(13,5)))# appending towards the list of each visulization img
    return render_template("stock_info.html",
                           stock_name = selected_stock.upper(),
                           described_data=described_data,
                           plot_base64=plot_list,
                           ml_algorithm=ml_algorithm,
                           predict_plot=plotted,
                           rmse_a=rmse
                           )

@app.route('/about',methods=['GET'])
def about_team():
    return render_template("team.html")

if __name__ == '__main__':
    app.run(debug=True,port=8080)
