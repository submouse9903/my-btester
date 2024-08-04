import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from myBtester import Strategy, Backtest
from lightweight_charts import Chart
import talib
import yfinance as yf
import pandas as pd
import requests
import time


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
class MACrossoverStrategy_RiskManagement_ATR(Strategy):

    def init(self, fast_period: int, slow_period: int):
        self.max_total_unit_risk = .01  # 거래당 1% 리스크 지는 것
        self.highpoint_of_account = 0  # 초기 현금으로 최대 계좌 초기화
        self.account_pole = self.cash  # 자산 최대값 기준로 리스크단위 증감 계산하기 위함 초기 계좌기준 10%씩 늘어나면 갱신하는 구조로할까

        self.fast_ma = {}  # next에서 가격데이터랑 같은 날짜로 반복문이 돌려진다.
        self.slow_ma = {}
        self.atr = {}
        df = pd.DataFrame()  # Empty DataFrame

        for symbol in self.symbols:
            self.fast_ma[symbol] = talib.EMA(self.data[(symbol, 'Close')], timeperiod=fast_period)
            self.slow_ma[symbol] = talib.EMA(self.data[(symbol, 'Close')], timeperiod=slow_period)
            self.atr[symbol] = talib.ATR(self.data[(symbol, 'High')], self.data[(symbol, 'Low')],self.data[(symbol, 'Close')], timeperiod=20)

    def next(self, i, record):  # 지표까지 모두 준비된 데이터에 대해서 모두 동일한 시점에서 시작하는 부분이다. 35,285.57%
        # i는 이동하는 캔들이다. 계속 캔들을 갱신하는 것이다, 지표는 앞전에 init함수로 미리 데이터를 만들어 놓고, 이 부분에서 하나씩 갱신하는 것.
        self.riskSize()

        for symbol in self.symbols:
            open_condition1 = self.fast_ma[symbol][i - 2] < self.slow_ma[symbol][i - 2]
            open_condition2 = self.fast_ma[symbol][i - 1] > self.slow_ma[symbol][i - 1]
            if open_condition1 and open_condition2 and self.atr[symbol][i] > 0:  # 해당 종목에서 포지션이 있더라도 조전에 만족하면 그냥 진입함. 피라미딩같은 느낌
                position_size = self.positionSize(record[(symbol, 'Open')], self.atr[symbol][i])
                stop_loss = record[(symbol, 'Open')] - self.atr[symbol][i]
                self.open(symbol=symbol, price=record[(symbol, 'Open')], size=position_size, stopLoss=stop_loss, long=True)

        for position in self.open_positions[:]:  # [:]리스트 맨 앞부터 맨 뒤까지 불러오는 슬라이싱, i가 돌면서 매 순간마다 포지션을 수시로 불러오는 것다. 그리고 조건이 되면 close를 하는 것이고
            # 매 순간마다 포지션을 모두 불러오고 해당 순간의 가격에 대해서 조건이 맞으면 close를 하는 로직인데.. 처음 진입 시점의 atr을 알아야된다고...
            close_condition1 = self.fast_ma[position.symbol][i - 1] < self.slow_ma[position.symbol][i - 1]
            stoploss_condition1 = position.stopLoss > record[(position.symbol, 'Open')]
            if close_condition1 or stoploss_condition1:  # atr만이 청산전략이 아니다. 다른 가격적인 조건이 청산 조건이 추가되어야함. 청산에서 atr은 정해진 리스크량으로 정의하기 위해서 사용된 것
                self.close(position=position, price=record[(position.symbol, 'Open')])

    def positionSize(self, price: float, atr: float):
        if pd.isna(self.cash) or pd.isna(self.assets_value) or pd.isna(self.max_total_unit_risk) or pd.isna(atr): return 0
        return round((self.cash + self.assets_value) * self.max_total_unit_risk / (2 * atr)) if price > 0 else 0

    def riskSize(self):
        self.highpoint_of_account = max(self.highpoint_of_account, self.cash + self.assets_value)
        if self.highpoint_of_account > self.account_pole * 1.11: self.account_pole *= 1.11
        # self.account_pole = self.highpoint_of_account
        condition1 = self.cash + self.assets_value < self.account_pole * (1 - 0.11)
        condition2 = self.cash + self.assets_value < self.account_pole * (1 - 0.22)
        condition3 = self.cash + self.assets_value < self.account_pole * (1 - 0.33)
        condition1_2 = self.cash + self.assets_value > self.account_pole * (1 - 0.22)
        condition2_3 = self.cash + self.assets_value > self.account_pole * (1 - 0.33)
        condition3_4 = self.cash + self.assets_value > self.account_pole * (1 - 0.44)
        if condition1 and condition1_2: self.max_total_unit_risk = .01 * 0.8 * (0.8 * 0.8 * 0.8 * 0.8)
        if condition2 and condition2_3: self.max_total_unit_risk = .01 * 0.8 * 0.8 * (0.8 * 0.8 * 0.8 * 0.8)
        if condition3 and condition3_4: self.max_total_unit_risk = .01 * 0.8 * 0.8 * 0.8 * (0.8 * 0.8 * 0.8 * 0.8)
        if not condition1: self.max_total_unit_risk = .01  # 다시 자산이 정상화되면 리스크 단위도 정상화

def getStratStats(log_returns: pd.Series, risk_free_rate: float = 0.02):
    stats = {}  # Total Returns
    stats['tot_returns'] = np.exp(log_returns.sum()) - 1
    # Mean Annual Returns
    stats['annual_returns'] = np.exp(log_returns.mean() * 252) - 1
    # Annual Volatility
    stats['annual_volatility'] = log_returns.std() * np.sqrt(252)
    # Sortino Ratio
    annualized_downside = log_returns.loc[log_returns < 0].std() * np.sqrt(252)
    stats['sortino_ratio'] = (stats['annual_returns'] - risk_free_rate) / annualized_downside
    # Sharpe Ratio
    stats['sharpe_ratio'] = (stats['annual_returns'] - risk_free_rate) / stats['annual_volatility']

    # Max Drawdown
    cum_returns = log_returns.cumsum() - 1
    peak = cum_returns.cummax()
    drawdown = peak - cum_returns
    max_idx = drawdown.argmax()
    stats['max_drawdown'] = 1 - np.exp(cum_returns.iloc[max_idx])/ np.exp(peak.iloc[max_idx])

    # Max Drawdown Duration
    strat_dd = drawdown[drawdown == 0]
    strat_dd_diff = strat_dd.index[1:] - strat_dd.index[:-1]
    strat_dd_days = strat_dd_diff.map(lambda x: x.days).values
    strat_dd_days = np.hstack([strat_dd_days,(drawdown.index[-1] - strat_dd.index[-1]).days])
    stats['max_drawdown_duration'] = strat_dd_days.max()
    return {k: np.round(v, 4) if type(v) == np.float_ else v
            for k, v in stats.items()}
# print(system.get_transactions())

def get_bar_data(symbol):
    # symbol_ = selected_rows[selected_rows == symbol]
    print(symbol)
    return global_system_data[symbol] #전역 변수를 인지할 수 있나

def on_search(chart, searched_string):  # Called when the user searches.
    new_data = get_bar_data(searched_string)
    if new_data.empty:
        return
    chart.topbar['symbol'].set(searched_string)
    chart.set(new_data)

def calculate_sma(df, period):
    return pd.DataFrame({
        'time': df.index,
        f'EMA {period}': talib.EMA(df['Close'], timeperiod=period) #df['Close'].rolling(window=period).mean()
    }).dropna()


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
    line_slow.set(calculate_sma(new_data, period=60))
    line_fast.set(calculate_sma(new_data, period=20))
    chart.set(new_data, True)
    chart.marker_list(trade_history)
    chart.watermark(chart.topbar['symbol'].value, color='rgba(100, 100, 100, 0.3)')


if __name__ == '__main__':
    chart = Chart(toolbox=True)
    line_slow = chart.create_line('EMA 60', color= 'rgba(214, 237, 255, 0.7)')
    line_fast = chart.create_line('EMA 20', color= 'rgba(255, 100, 100, 0.6)')


    start = '2000-01-01'
    end = '2024-07-01'
    assets = 'nasdaq100'
    benchmark = 'NQ=F'

    ldf = pd.read_html(requests.get(f'https://www.slickcharts.com/{assets}', headers={'User-agent': 'Mozilla/5.0'}).text)
    symbols = [x.replace('.', '-') for x in ldf[0]['Symbol'] if isinstance(x, str)]
    downloads = yf.download([*symbols, benchmark], start, end, group_by='ticker')
    data = downloads[symbols]
    benchmark = downloads[benchmark]['Close']
    data2 = data.fillna(method='ffill', limit=1)

    tickers = symbols
    selected_rows = tickers

    start = time.time()
    backtest = Backtest(MACrossoverStrategy_RiskManagement_ATR, data2, commission=.003, cash=1e3)
    result = backtest.run(20, 60)


    import quantstats as qs

    temt2 = qs.plots.log_returns(result.returns, benchmark, fontname='sans-serif')
    temt3 = qs.plots.drawdown(result.returns, figsize=(10, 3), fontname='sans-serif')

    temt1 = qs.reports.html(result.returns, benchmark, output='./file-name.html')







    global_system_data = backtest.data
    global_trade_history_df = trades_to_dataframe(result.trades)

    end = time.time()
    print(f"{end - start:.5f} sec")
    chart.legend(True)
    chart.events.search += on_search
    chart.topbar.switcher('symbol', tuple(selected_rows), default=tuple(selected_rows)[0], func=on_security_selection)
    df = backtest.data[tuple(selected_rows)[0]]
    line_slow.set(calculate_sma(df, period=60))
    line_fast.set(calculate_sma(df, period=20))
    chart.set(df)
    chart.show(block=True)
