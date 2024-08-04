import pandas as pd
from lightweight_charts import Chart


def get_bar_data(symbol, timeframe):
    if symbol not in ('AAPL', 'GOOGL', 'TSLA'):
        print(f'No data for "{symbol}"')
        return pd.DataFrame()
    return pd.read_csv(f'bar_data/{symbol}_{timeframe}.csv')


def on_search(chart, searched_string):  # Called when the user searches.
    new_data = get_bar_data(searched_string, chart.topbar['timeframe'].value)
    if new_data.empty:
        return
    chart.topbar['symbol'].set(searched_string)
    chart.set(new_data)

def on_security_selection(chart):  # Called when the user changes the timeframe.
    new_data = get_bar_data(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
    if new_data.empty:
        return
    chart.set(new_data, True)

def on_timeframe_selection(chart):  # Called when the user changes the timeframe.
    new_data = get_bar_data(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
    if new_data.empty:return
    chart.set(new_data, True)

# def on_trade_history(chart, trades):  # Called when the user changes the timeframe.


def on_horizontal_line_move(chart, line):
    print(f'Horizontal line moved to: {line.price}')
def calculate_sma(df, period: int = 50):
    return pd.DataFrame({
        'time': df['date'],
        f'SMA {period}': df['close'].rolling(window=period).mean()
    }).dropna()


import yfinance as yf
import pandas as pd
import requests

start = '2000-01-01'
end = '2024-07-01'

assets = 'nasdaq100'
benchmark = 'NQ=F'

ldf = pd.read_html(requests.get(f'https://www.slickcharts.com/{assets}', headers={'User-agent': 'Mozilla/5.0'}).text)
symbols = [x.replace('.','-') for x in ldf[0]['Symbol'] if isinstance(x, str)]
downloads = yf.download([*symbols, benchmark], start, end, group_by='ticker')

data = downloads[symbols]
data
# data.set_index(data.DatetimeIndex(data["Date"]), inplace=True)
benchmark = downloads[benchmark]['Close']

if __name__ == '__main__':
    chart = Chart(toolbox=True)
    chart.legend(True)

    chart.events.search += on_search
    #test = ('TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'TSM', 'NVDA', 'JPM', 'JNJ')
    chart.topbar.switcher('symbol', ('TSLA', 'qwer', 'GOOGL'), default='TSLA',func=on_security_selection)
    chart.topbar.switcher('timeframe', ('1min', '5min', '30min'), default='5min', func=on_timeframe_selection)

    df = get_bar_data('TSLA', '5min')

    chart.set(df)



    chart.horizontal_line(200, func=on_horizontal_line_move)

    chart.show(block=True)
