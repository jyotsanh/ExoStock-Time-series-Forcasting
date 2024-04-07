import numpy
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy
import pandas as pd
import os

df = pd.read_csv("./STOCKS/bnl.csv")

#candle stick patter
def candle_stick_chart():
    return None

#-----------------checked-below--------------------------------------
def scales_the_data(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    scale_data = scaler.fit_transform(df)
    return scale_data,scaler


def preprocessing(data):
    x_data = []
    y_data = []
    for i in range(100,len(data)):
        x_data.append(data[i-100:i])
        y_data.append(data[i])
    return numpy.array(x_data),numpy.array(y_data)


def rmse(orginal,prediciton):
    rms = numpy.sqrt(numpy.mean((orginal-prediciton)**2))
    return rms

def LSTM_model(X_train, X_test, y_train, y_test, scaler):


    model = Sequential()
    model.add(LSTM(4,input_shape=(X_train.shape[1],1)))

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss= 'mean_squared_error',
    )

    model.fit(X_train,y_train,batch_size = 1,epochs=2)
    print(model.summary())

    # predict the stock using test data.
    predictions = model.predict(X_test)
    inv_p = scaler.inverse_transform(predictions)
    print('--------------------------')
    print(f"Prediction : \n {inv_p}")
    inv_y = scaler.inverse_transform(y_test)
    print('--------------------------')
    print(f"Original : \n {inv_y}")

    mse = rmse(inv_y,inv_p)

    return mse,inv_p,inv_y



def train_model(stock_name,algorithim):
    df = pd.read_csv(f"./STOCKS/{stock_name}.csv")
    close_price = df[['close']]

    #scaled the data through Min-Max Scaler.
    scaled_data,scaler = scales_the_data(close_price)

    #convert the data into numpy x,y
    x_data,y_data = preprocessing(scaled_data)

    #split the data for train and testing
    splitting_len = int(len(x_data)*0.7)
    X_train = x_data[:splitting_len]
    y_train = y_data[:splitting_len]
    X_test = x_data[splitting_len:]
    y_test = y_data[splitting_len:]
    print(f"Shapes {X_train.shape,y_train.shape,X_test.shape,y_test.shape}")
    
    mse,prediction,original = LSTM_model(X_train, X_test, y_train, y_test, scaler)

    predicted_df = pd.DataFrame(
        {
            'original_data':original.reshape(-1),
            'prediction':prediction.reshape(-1),
            'date':df.date[splitting_len+100:]
                            }
    )
    predicted_df = predicted_df.sort_values('date')
    predicted_df['date'] = pd.to_datetime(predicted_df['date'])
    # Set 'date' column as index
    predicted_df.set_index('date', inplace=True)
    print(predicted_df)
    plt.figure(figsize=(14,7))
    plt.plot(predicted_df['original_data'],label="original price")
    plt.plot(predicted_df['prediction'],label="predicted price")

    plt.title('Actual Price vs Predicted Price over time')

   
    plt.grid(True)
    plt.xlabel('years')
    plt.ylabel('price')
    plt.legend()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    return mse,plot_base64

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



#-----------------checked-above--------------------------------------
if __name__ == '__main__':
    dataframe_describe()
    list_stock_files()
    plot_function()
    LSTM_model()