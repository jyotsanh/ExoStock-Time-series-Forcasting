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
from sklearn.linear_model import LinearRegression
import os

from bs4 import BeautifulSoup
import requests

df = pd.read_csv("./STOCKS/bnl.csv")

#candle stick patter
def candle_stick_chart():
    return None

#-----------------checked-below--------------------------------------


def stock_details(stock_name):
    url = f"https://www.sharesansar.com/company/{stock_name.lower()}"

    # Fetch HTML content from the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    print(f"response code : {response.status_code}")
    if response.status_code == 200:
        # Parse the HTML content
    
        soup = BeautifulSoup(response.text, 'html.parser')
        print('*****************************')
        tables = soup.find_all('div',class_='tab-content') #finds the table from html
        table_data = tables[0].find_all('td') # finds the all table_data
        Symbol = table_data[1].text.strip()
        Name = table_data[3].text.strip()
        print('---------------------------')
        Sector = table_data[5].text.strip()
        Listed_Shares = table_data[7].text.strip()
        Paid_Up = table_data[9].text.strip()
        Operation_Date = table_data[11].text.strip()
        Phone_Number = table_data[13].text.strip() 
        Email = table_data[15].text.strip()
        Address = table_data[17].text.strip()
        
        data = {
            'Symbol':Symbol,
            'Name':Name,
            'Sector':Sector,
            'Listed_Share':Listed_Shares,
            'Paid up': Paid_Up,
            'Operation_date':Operation_Date,
            'Phone_Number':Phone_Number,
            'Email':Email,
            'Address':Address
        }
        
        return data
    else:
        print("Failed to fetch HTML:", response.status_code)
        return False


def scales_the_data(df):
    scaler = MinMaxScaler(feature_range=(0,1)) #Min-Max Scaler is used to scale the data
    scale_data = scaler.fit_transform(df) # fit-trandform the data in between 0-1
    return scale_data,scaler #return both scaler and scaled_data for inverse_transform.


def preprocessing(data):
    x_data = []  # list for x_data
    y_data = []  # list for y_data
    for i in range(100,len(data)): 
        # 100 was used because we will prediict stock data taking data 100 from before.
        x_data.append(data[i-100:i]) # appends 100 data into x_data
        y_data.append(data[i]) # target data
    return numpy.array(x_data),numpy.array(y_data) #returns numpy list.


def rmse(orginal,prediciton): # root-mean-square evaluation method.
    rms = numpy.sqrt(numpy.mean((orginal-prediciton)**2)) #formula.
    return rms #return rmse.

def LSTM_model(X_train, X_test, y_train, y_test, scaler):
    # Lstm model takes 4 parameters X_train, X_test, y_train, y_test, scaler

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

    inv_p = scaler.inverse_transform(predictions) #inverse the predicted data into normal form.

    inv_y = scaler.inverse_transform(y_test) # inverse the y_test data into normal form.

    root_mse = rmse(inv_y,inv_p)  #finding the root mean square between predicted and original data

    return root_mse,inv_p,inv_y

def linear_model(X_train, X_test, y_train, y_test, scaler):

    model = LinearRegression()
    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    X_test_flattened = X_test.reshape(X_test.shape[0], -1)

    # Now X_train_flattened and X_test_flattened should have shape (630, 100*1) = (630, 100)

    # Then, you can fit the model
    model.fit(X_train_flattened, y_train)
    
    predictions = model.predict(X_test_flattened)

    inv_p = scaler.inverse_transform(predictions)
    
    inv_y = scaler.inverse_transform(y_test)


    mse = rmse(inv_y,inv_p)

    return mse,inv_p,inv_y

def final(original,prediction,df,splitting_len):
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
   
    plt.figure(figsize=(15,6))
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
    return plot_base64

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

    
    if algorithim=='Linear Regression':
        mse,prediction,original = linear_model(X_train, X_test, y_train, y_test, scaler)
        plot_base64 = final(original,prediction,df,splitting_len)
        return mse,plot_base64
    
    elif algorithim=='LSTM':
        mse,prediction,original = LSTM_model(X_train, X_test, y_train, y_test, scaler)
        plot_base64 = final(original,prediction,df,splitting_len)
        return mse,plot_base64
    else:
        return False,False

    

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
        
        return
    
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out only CSV files
    csv_files = [file for file in files if file.endswith('.csv')]
    
    # Print the list of stock files
    if csv_files:
        stock_names = []
        
        for file in csv_files:
            stock_names.append(file.replace(".csv",""))
        return stock_names
    else:
        print("No CSV files found in the folder.")



#-----------------checked-above--------------------------------------
if __name__ == '__main__':
    list_stock_files()
    plot_function()
    dataframe_describe()
    preprocessing()
    train_model()
    linear_model()
    LSTM_model()
    scales_the_data()
    rmse()