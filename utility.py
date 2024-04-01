import numpy
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy
import pandas as pd
import math
from sklearn.linear_model import ElasticNet
import os

df = pd.read_csv("./STOCKS/bnl.csv")

#candle stick patter
def candle_stick_chart():
    return None

#-----------------checked-below--------------------------------------
def dataframe_describe(stock_name):
    df = pd.read_csv(f"./STOCKS/{stock_name}.csv")
    df_describe_html = df.describe().to_html()
    return df_describe_html

# Generate plot dynamically

def plot_function(stock_name,column_name,fig=(15,6)):
    described_data = pd.read_csv(f"./STOCKS/{stock_name}.csv")
    
    # Convert 'date' column to datetime format
    described_data['date'] = pd.to_datetime(described_data['date'])
    # Set 'date' column as index
    described_data.set_index('date', inplace=True)
    
    # Generate plot
    plt.figure(figsize=fig)
    described_data[column_name].plot()
    plt.xlabel('years')
    plt.ylabel(f'{column_name}')
    plt.title(f"{column_name} data for {stock_name} data")    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return plot_base64

# this function returns all stocks name in stocks_data folder
def list_stock_files(folder_path="./STOCKS"):
    # Check if the folder path exists

    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return
    
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out only CSV files
    csv_files = [file for file in files if file.endswith('.csv')]
    
    # Print the list of stock files
    if csv_files:
        stock_names = []
        print("List of stock files:")
        for file in csv_files:
            stock_names.append(file.replace(".csv",""))
        return stock_names
    else:
        print("No CSV files found in the folder.")



# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

def create_preprocessed_Dataset(df):
    df['Open'] = df['Open'].str.replace(',', '').astype(float)
    df['High'] = df['High'].str.replace(',', '').astype(float)
    df['Low'] = df['Low'].str.replace(',', '').astype(float)
    df['Ltp'] = df['Ltp'].str.replace(',', '').astype(float)
    # Drop columns except 'date' and 'open'
    df.drop(columns=df.columns.difference(['Date', 'Open']), inplace=True)

    df = df['Open']
    # only use Open column
    dataset = df.values
    
    dataset = dataset.reshape(-1, 1)
    
    dataset = dataset.astype('float32')
    
    # split into train and test sets
    train_size = len(dataset) - 2
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    # trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    # testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    return trainX, trainY, testX, testY



#-----------------checked-above--------------------------------------

def getData(df):
    # Create the lists / X and Y data sets
    dates = []
    prices = []

    # Get the number of rows and columns in the data set
    # df.shape

    # Get the last row of data (this will be the data that we test on)
    last_row = df.tail(1)

    # Get all of the data except for the last row
    df = df.head(len(df) - 1)
    # df

    # The new shape of the data
    # df.shape

    # Get all of the rows from the Date Column
    df_dates = df.loc[:, 'date']
    # Get all of the rows from the Open Column
    df_open = df.loc[:, 'open']

    # Create the independent data set X
    for date in df_dates:
        dates.append([int(date.split('-')[2])])

    # Create the dependent data se 'y'
    for open_price in df_open:
        prices.append(float(open_price))

    # See what days were recorded
    last_date = int(((list(last_row['date']))[0]).split('-')[2])
    last_price = float((list(last_row['open']))[0])
    return dates, prices, last_date, last_price


def LSTM_model(dates, prices, test_date, df):
    df.drop(df.columns.difference(['date', 'open']), 1, inplace=True)
    df = df['open']
    dataset = df.values
    dataset = dataset.reshape(-1, 1)
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = len(dataset) - 2
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
    # reshape input to be [samples, time steps, features]
    X_train = numpy.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = numpy.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

    # make predictions
    trainPredict = model.predict(X_train)
    mainTestPredict = model.predict(X_test)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    y_train = scaler.inverse_transform([y_train])

    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    mainTestPredict = scaler.inverse_transform(mainTestPredict)
    mainTestPredict = [item for sublist in mainTestPredict for item in sublist]
    y_test = scaler.inverse_transform([y_test])
    test_score = mean_squared_error(y_test[0], mainTestPredict)
    # calculate root mean squared error
    trainPredict = [item for sublist in trainPredict for item in sublist]

    # print(trainPredict, testPredict[0])

    return (trainPredict, (testPredict[0])[0], test_score)

def main():
    return 0



if __name__ == '__main__':
    main()
    dataframe_describe()
    list_stock_files()
    plot_function()
    LSTM_model()