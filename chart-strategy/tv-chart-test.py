import matplotlib.pyplot as plt
import numpy as np
from myBtester import Backtest
from btester_strategies import MACrossoverStrategy_RiskManagement_ATR, MACrossoverStrategy_RiskManagement_ATR_covariance
from lightweight_charts import Chart
import talib
import yfinance as yf
import pandas as pd
import requests
import time
import FinanceDataReader as fdr

def trades_to_dataframe(trades: list) -> pd.DataFrame:
    # Extract attributes from each Trade object and store them in a dictionary
    data = {
        'symbol': [trade.symbol for trade in trades],
        'open_date': [trade.open_date for trade in trades],
        'close_date': [trade.close_date for trade in trades],
        'open_price': [trade.open_price for trade in trades],
        'close_price': [trade.close_price for trade in trades],
        'position_size': [trade.position_size for trade in trades],
        'profit_loss': [trade.profit_loss for trade in trades],
        'change_pct': [trade.change_pct for trade in trades],
        'trade_commission': [trade.trade_commission for trade in trades],
        'cumulative_return': [trade.cumulative_return for trade in trades],
        'long': [trade.long for trade in trades]
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)
    return df

def get_bar_data(symbol):
    print(symbol)
    return global_system_data[symbol] #전역 변수를 인지할 수 있나

def on_search(chart, searched_string):  # Called when the user searches.
    new_data = get_bar_data(searched_string)
    if new_data.empty: return
    chart.topbar['symbol'].set(searched_string)
    chart.set(new_data)

def calculate_ema(df, period): return pd.DataFrame({'time': df.index, f'EMA {period}': talib.EMA(df['Close'], timeperiod=period)}).dropna()
def calculate_sma(df, period): return pd.DataFrame({'time': df.index, f'SMA {period}': talib.SMA(df['Close'], timeperiod=period)}).dropna()

def get_trade_history(symbol):
    # trade_history = [
    #     {"time": "2021-01-21", "position": "above", "shape": "arrow_down", "color": "#FF0000", "text": "s"},
    #     {"time": "2022-01-22", "position": "below", "shape": "arrow_up", "color": "#2196F3", "text": "b"}
    # ]
    trade_history = []
    counter = 0
    for index, row in global_trade_history_df.loc[global_trade_history_df['symbol'] == symbol].iterrows():
        trade_open = {
                "time": row['open_date'].strftime('%Y-%m-%d'),
                "position": "below",
                "shape": "arrow_up",
                "color": "#2196F3",
                "text": str(round(row['position_size'], 1))
            }
        trade_history.append(trade_open)
        trade_close = {
                "time": row['close_date'].strftime('%Y-%m-%d'),
                "position": "above",
                "shape": "arrow_down",
                "color": "#00FF00" if round(row['profit_loss'], 1) > 0 else "#FF0000",
                "text": str(round(row['profit_loss'], 1))
            }
        trade_history.append(trade_close)
        counter += 1
    return trade_history
def on_security_selection(chart):  # Called when the user changes the timeframe.
    new_data = get_bar_data(chart.topbar['symbol'].value)
    trade_history = get_trade_history(chart.topbar['symbol'].value)
    if new_data.empty: return
    # chart.marker(text='S', time='2022-5-22', position='inside', shape='arrow_down', color='#FF0000', )

    chart.clear_markers() #심볼을 바꾸면서 이전 심볼이랑 겹치지 않게 만들어줘야함
    line_slow.set(calculate_ema(new_data, period=60))
    line_fast.set(calculate_ema(new_data, period=20))
    line_200.set(calculate_sma(new_data, period=200))

    chart.set(new_data, True)
    chart.marker_list(trade_history)
    chart.watermark(chart.topbar['symbol'].value, color='rgba(100, 100, 100, 0.3)')

def place_buy_order(key):
    print(f'Buy {key} shares.')
def place_sell_order(key):
    print(f'Sell all shares, because I pressed {key}.')

if __name__ == '__main__':
    chart = Chart(toolbox=True)
    chart.hotkey('shift', (1, 2, 3), place_buy_order)
    chart.hotkey('shift', 'X', place_sell_order)
    line_slow = chart.create_line('EMA 60', color= 'rgba(214, 237, 255, 0.7)')
    line_fast = chart.create_line('EMA 20', color= 'rgba(255, 100, 100, 0.6)')
    line_200 = chart.create_line('SMA 200', color= 'rgba(255, 100, 100, 0.6)')

    start = '2000-01-01'
    end = '2024-07-01'

    # tickers_NASDAQ = fdr.StockListing('NASDAQ')  # 나스닥 (NASDAQ): 4천+ 종목
    # tickers_NYSE = fdr.StockListing('NYSE')  # 뉴욕증권거래소 (NYSE): 3천+ 종목
    tickers_kospi = fdr.StockListing('KOSPI')[['Code', 'Name']]  # 한국거래소: 7,632 종목
    # tickers_kosdaq = fdr.StockListing('KOSDAQ')[['Code', 'Name']]  # 한국거래소: 7,632 종목

    benchmark = pd.read_pickle('/Users/dongin/Desktop/my-btester/chart-strategy/stock-data-pickles/indicies/KOSPI(^KS11).pkl')['Close']
    symbols = tickers_kospi['Code'].values
    #
    # data = downloads[symbols]
    data = pd.read_pickle('/Users/dongin/Desktop/my-btester/chart-strategy/stock-data-pickles/kospi.pkl')
    data2 = data[symbols]
    data2.rename(columns={'Ticker': 'symbol'}, inplace=True)
    # data.set_index(data.DatetimeIndex(data["Date"]), inplace=True)


    selected_rows = symbols

    start = time.time()
    backtest = Backtest(MACrossoverStrategy_RiskManagement_ATR, data2, commission=.003, cash=5*1e6) # 미국 3*1e3, 한국: 5*1e6
    # backtest = Backtest(MACrossoverStrategy_RiskManagement_ATR_covariance, data2, commission=.003, cash=1e3)

    result = backtest.run(20, 60)


    import quantstats as qs
    # temt2 = qs.plots.log_returns(result.returns, benchmark, fontname='sans-serif')
    # temt3 = qs.plots.drawdown(result.returns, figsize=(10, 3), fontname='sans-serif')
    temt1 = qs.reports.html(result.returns, benchmark, output='./file-name.html')







    global_system_data = backtest.data
    global_trade_history_df = trades_to_dataframe(result.trades)

    end = time.time()
    print(f"{end - start:.5f} sec")
    chart.legend(True)
    chart.events.search += on_search
    chart.topbar.switcher('symbol', tuple(selected_rows), default=tuple(selected_rows)[0], func=on_security_selection)
    df = backtest.data[tuple(selected_rows)[0]]
    line_slow.set(calculate_ema(df, period=60))
    line_fast.set(calculate_ema(df, period=20))
    line_200.set(calculate_sma(df, period=200))

    chart.set(df)
    chart.show(block=True)
