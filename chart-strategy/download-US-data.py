import yfinance as yf
import pandas as pd
import requests
import FinanceDataReader as fdr

start = '2000-01-01'
end = '2024-07-01'

assets = 'nasdaq100'
benchmark = 'NQ=F'

NASDAQ = fdr.StockListing('NASDAQ') # 나스닥 (NASDAQ): 4천+ 종목
NYSE = fdr.StockListing('NYSE') # 뉴욕증권거래소 (NYSE): 3천+ 종목

# ldf = pd.read_html(requests.get(f'https://www.slickcharts.com/{assets}', headers={'User-agent': 'Mozilla/5.0'}).text)
# symbols = [x.replace('.','-') for x in ldf[0]['Symbol'] if isinstance(x, str)]

downloads_NASDAQ = yf.download([*NYSE['Symbol'].values, benchmark], start, end, group_by='ticker', threads=False)
# downloads_NYSE = yf.download([*NASDAQ['Symbol'].values, benchmark], start, end, group_by='ticker', threads=False)
