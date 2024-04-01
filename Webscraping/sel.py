from bs4 import BeautifulSoup      # -> beautiful soup help to webscrap html page
from selenium import webdriver     # -> webdriver helps to automate webscrap
import time                        # -> time library helps to skip
from selenium.webdriver.common.by import By
import pandas as pd
import os

options = webdriver.FirefoxOptions()  # -> This allow to access the firefox web browser
options.add_argument("-headless")   # -> This run browser in background

stock_folder_path = "../stock_datas"

driver = webdriver.Firefox(options=options) # ->it provide  argument to driver which run browser in background


# this function returns all stocks name in stocks_data folder
def list_stock_files(folder_path):
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


#  -> Below function take 'dictionary' argument and returns pandas dataframe
def dictionary_to_dataframe(dictionary:dict):
    
    return pd.DataFrame(dictionary)
#  -> Below function takes 'page html' as argument and returns pandas dataframe
def page_source_to_dataframe(page_html):
    soup = BeautifulSoup(page_html, 'html.parser')
    one = soup.find_all('div',class_ = 'col-md-10')
    tr = one[1].find_all('tr')
    # -> Create a list to save data for each column
    index = []
    date = []
    Open = []
    High = []
    Low = []
    Ltp = []
    per_change = []
    Quantity = []
    turnover = []
    for i in range(12,62): #from list of table row => 'tr' the index -> 1 starts from 12 to 62 which is 50 rows of data
        index.append(tr[i].find_all('td')[0].text.strip())
        date.append(tr[i].find_all('td')[1].text.strip())
        Open.append(tr[i].find_all('td')[2].text.strip())
        High.append(tr[i].find_all('td')[3].text.strip())
        Low.append(tr[i].find_all('td')[4].text.strip())
        Ltp.append(tr[i].find_all('td')[5].text.strip())
        per_change.append(tr[i].find_all('td')[6].text.strip())
        Quantity.append(tr[i].find_all('td')[7].text.strip())
        turnover.append(tr[i].find_all('td')[8].text.strip())


    df = dictionary_to_dataframe({
        'date':date,
        'open':Open,
        'high':High,
        'low':Low,
        'ltp':Ltp,
        '% change':per_change,
        'turnover':turnover
    })

    return df

# in the below list provide the stock name for example : 'lbl','sanima' which you want to scrape
stock_list = ['lsl']
for stock_name in stock_list:

    stock_list = list_stock_files(stock_folder_path) # -> list all the files in stocks_data folder.
    if stock_name in stock_list:# ->checks if the stocks is already in folder.
        print(f"{stock_name} is already in folder {stock_folder_path}")
    else:
        try:# try to get the url of the stock if it's valid.

            #  -> accesing nabil stock from sharesansar website
            driver.get(f"https://www.sharesansar.com/company/{stock_name}")
            
            # -> 'price history' button x_path which driver uses to click
            price_history_xpath = "/html/body/div[2]/div/section[2]/div[3]/div/div/div/div[2]/div/div[1]/div[1]/ul/li[8]"
            button = driver.find_element(By.XPATH,price_history_xpath)
            # -> driver click button
            button.click()
            
            # -> Locate the select dropdown option using XPath
            fifty_xpath = "/html/body/div[2]/div/section[2]/div[3]/div/div/div/div[2]/div/div[1]/div[2]/div/div[8]/div/div/div[1]/label/select/option[3]"
            button = driver.find_element(By.XPATH,fifty_xpath)
            button.click()
            time.sleep(3)

            # Initialize an empty list to store DataFrames
            dfs = []

            # -> Loop 20 times to click on next button/link
            for _ in range(20):

                try:# -> some stocks has less than 20 pages for that we need to handle the error.
                    page_html = driver.page_source
                    stock_df = page_source_to_dataframe(page_html=page_html)
                    next_path = "/html/body/div[2]/div/section[2]/div[3]/div/div/div/div[2]/div/div[1]/div[2]/div/div[8]/div/div/div[5]/a[2]"
                    button = driver.find_element(By.XPATH,next_path)
                    button.click()
                    time.sleep(4)
                    # Get the HTML content of the page after clicking the button
                    
                    print(_)
                    # Append DataFrame to the list
                    dfs.append(stock_df)
                except:
                    print(f"error at {_}")
                

            # Concatenate all DataFrames in the list
            df = pd.concat(dfs, ignore_index=True)
            df['open'] = df['open'].str.replace(',', '').astype(float)
            df['high'] = df['high'].str.replace(',', '').astype(float)
            df['low'] = df['low'].str.replace(',', '').astype(float)
            df['ltp'] = df['ltp'].str.replace(',', '').astype(float)
            df['turnover'] = df['turnover'].str.replace(',', '').astype(float)
            # Save concatenated DataFrame to a single CSV file without index
            df.to_csv(f'../stock_datas/{stock_name}.csv', index=False)
            print(f"Concatenated DataFrame saved to {stock_name}.csv")
        except:
            print(f"{stock_name} doesn't exist or driver can't get the url")
            
driver.quit()