import yfinance as yf
import pandas as pd
import requests
import FinanceDataReader as fdr

start = '2000-01-01'
end = '2024-07-01'

assets = 'nasdaq100'
benchmark = 'NQ=F'

ldf = pd.read_html(requests.get(f'https://www.slickcharts.com/{assets}', headers={'User-agent': 'Mozilla/5.0'}).text)
symbols = [x.replace('.','-') for x in ldf[0]['Symbol'] if isinstance(x, str)]

NASDAQ = fdr.StockListing('NASDAQ') # 나스닥 (NASDAQ): 4천+ 종목
NYSE = fdr.StockListing('NYSE') # 뉴욕증권거래소 (NYSE): 3천+ 종목

# df = fdr.DataReader('KRX-DELISTING:036360') # 3SOFT(036360)  # KRX delisting stock data 상장폐지 종목 전체 가격 데이터 이거 안됨
KOSPI = fdr.StockListing('KOSPI') # KOSPI: 940 종목
KOSDAQ = fdr.StockListing('KOSDAQ') # KOSDAQ: 1,597 종목
etfs = fdr.StockListing('ETF/KR') # 한국 ETF 전종목
# df = fdr.SnapDataReader('KRX/INDEX/LIST') # KRX 전체 지수목록 이거 안됨

SSE = fdr.StockListing('SSE') # 상하이 증권거래소 (Shanghai Stock Exchange: SSE): 1천+ 종목
SZSE = fdr.StockListing('SZSE') # 선전 증권거래소(Shenzhen Stock Exchange: SZSE): 1천+ 종목
HKEX = fdr.StockListing('HKEX') # 홍콩 증권거래소(Hong Kong Exchange: HKEX): 2천5백+ 종목
TSE = fdr.StockListing('TSE') # 도쿄 증권거래소(Tokyo Stock Exchange: TSE): 3천9백+ 종목
HOSE = fdr.StockListing('HOSE') # 호찌민 증권거래소(Ho Chi Minh City Stock Exchange: HOSE): 4백+ 종목



downloads = yf.download([*symbols, benchmark], start, end, group_by='ticker')

data = downloads
# data.set_index(data.DatetimeIndex(data["Date"]), inplace=True)
# benchmark = downloads[benchmark]['Close']

data2 = data.fillna(method='ffill', limit=1)

# data2.to_pickle('/Users/dongin/Desktop/my-btester/chart-strategy/yf_nasdaq100_000101-240701.pkl')