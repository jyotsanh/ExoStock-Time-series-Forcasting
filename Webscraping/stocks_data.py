from bs4 import BeautifulSoup
import requests

url = "https://www.sharesansar.com/company/gbime"

# Fetch HTML content from the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all('div',class_='tab-content') #finds the table from html
    table_data = tables[0].find_all('td') # finds the all table_data
    Symbol = table_data[1].text.strip()
    Name = table_data[3].text.strip()
    Sector = table_data[5].text.strip()
    Listed_Shares = table_data[7].text.strip()
    Paid_Up = table_data[9].text.strip()
    Operation_Date = table_data[11].text.strip()
    Phone_Number = table_data[13].text.strip() 
    Email = table_data[15].text.strip()
    Address = table_data[17].text.strip()

else:
    print("Failed to fetch HTML:", response.status_code)
