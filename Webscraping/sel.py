from bs4 import BeautifulSoup      # -> beautiful soup help to webscrap html page
from selenium import webdriver     # -> webdriver helps to automate webscrap
import time                        # -> time library helps to skip
from selenium.webdriver.common.by import By
import pandas as pd


options = webdriver.FirefoxOptions()  # -> This allow to access the firefox web browser

options.add_argument("-headless")   # -> This run browser in background


driver = webdriver.Firefox(options=options) # ->it provide  argument to driver which run browser in backgroung
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
        'Index':index,
        'Date':date,
        'Open':Open,
        'High':High,
        'Low':Low,
        'Ltp':Ltp,
        '% change':per_change,
        'turnover':turnover
    })

    return df


#  -> accesing nabil stock from sharesansar website
driver.get("https://www.sharesansar.com/company/nabil")

# -> 'price history' button x_path which driver uses to click
price_history_xpath = "/html/body/div[2]/div/section[2]/div[3]/div/div/div/div[2]/div/div[1]/div[1]/ul/li[8]"
button = driver.find_element(By.XPATH,price_history_xpath)
# -> driver click button
button.click()
# -> Locate the select dropdown option using XPath
fifty_xpath = "/html/body/div[2]/div/section[2]/div[3]/div/div/div/div[2]/div/div[1]/div[2]/div/div[8]/div/div/div[1]/label/select/option[3]"
button = driver.find_element(By.XPATH,fifty_xpath)
button.click()
time.sleep(1)

# Initialize an empty list to store DataFrames
dfs = []

# -> Loop 10 times to click on next button/link
for _ in range(25):
    page_html = driver.page_source
    df = page_source_to_dataframe(page_html=page_html)
    next_path = "/html/body/div[2]/div/section[2]/div[3]/div/div/div/div[2]/div/div[1]/div[2]/div/div[8]/div/div/div[5]/a[2]"
    button = driver.find_element(By.XPATH,next_path)
    button.click()
    time.sleep(2)
    # Get the HTML content of the page after clicking the button
    
    print(_)
    # Append DataFrame to the list
    dfs.append(df)

# Concatenate all DataFrames in the list
final_df = pd.concat(dfs, ignore_index=True)

# Save concatenated DataFrame to a single CSV file without index
final_df.to_csv('Nabil_Stock.csv', index=False)
print("Concatenated DataFrame saved to final_data.csv")

driver.quit()