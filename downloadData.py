'''
Download historical cryptocurrency market data to CSV format
'''

from time import sleep
from urllib.request import urlopen as u
from re import sub, findall

from bs4 import BeautifulSoup as b
import pandas as pd
from selenium import webdriver

def getData():
    df = pd.DataFrame(columns=('Currency', 'Date', 'Open', 'High', 'Low', 'Close', 'Value', 'Market Cap'))

    # Loop
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")

    driver = webdriver.Chrome(options=options)

    driver.get('https://coinmarketcap.com/coins/')
    for x in range(20):
        driver.execute_script(f'window.scrollTo(0, {x*500})')
        sleep(1)
    soup=b(driver.page_source,'html.parser')

    coins = soup.select('#__next > div.bywovg-1.sXmSU > div.main-content > div.sc-57oli2-0.comDeo.cmc-body-wrapper > div > div:nth-child(1) > div.h7vnx2-1.bFzXgL > table > tbody > tr > td:nth-child(3) > div > a > div > div > p')
    currencies = [x.text for x in coins]
    print(currencies)


    for currency in currencies:
        driver.get(f'https://coinmarketcap.com/currencies/{currency.replace(" ","-").lower()}/historical-data/')
        driver.execute_script('window.scrollTo(0, 1000)')
        sleep(1)
        for _ in range(50):
            driver.execute_script("document.querySelector('#__next > div > div.main-content > div.sc-57oli2-0.comDeo.cmc-body-wrapper > div > div.sc-16r8icm-0.jKrmxw.container > div > div > p:nth-child(3) > button').click()")
            sleep(2)
        soup=b(driver.page_source,'html.parser')
        coins = soup.select('#__next > div > div.main-content > div.sc-57oli2-0.comDeo.cmc-body-wrapper > div > div.sc-16r8icm-0.jKrmxw.container > div > div > div.b4d71h-2.hgYnhQ > table > tbody > tr')
        rows = [x.findChildren('td') for x in coins]
        for row in rows:
            myrow = [cell.text for cell in row]
            myrow[0] = pd.to_datetime(myrow[0])
            for x in range(1, len(myrow)):
                myrow[x] = float(myrow[x].replace('$','').replace(',',''))
            df.loc[len(df)] = [currency, *myrow]
            # print([cell.text for cell in row])  
        print(df)

    driver.quit()
    del driver

    df.to_json('historical_coin_data.json')
    df.to_csv('historical_coin_data.csv')

def testHvdInstallation():
    import horovod.spark.tensorflow as hvd
    from horovod.spark.common.store import HDFSStore
    import pyspark

    try:
        hvd.init()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    getData()
