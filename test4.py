import yfinance as yf
import pandas as pd
import requests
from myBtester import Strategy, Backtest
import quantstats as qs
import talib
# ㅁㄴㅇㄹasdf
import warnings
warnings.filterwarnings("ignore")

start = '2022-01-01'
end = '2024-01-01'

assets = 'nasdaq100'
benchmark = 'NQ=F'

ldf = pd.read_html(requests.get(f'https://www.slickcharts.com/{assets}', headers={'User-agent': 'Mozilla/5.0'}).text)
symbols = [x.replace('.','-') for x in ldf[0]['Symbol'] if isinstance(x, str)]
downloads = yf.download([*symbols, benchmark], start, end, group_by='ticker')

data = downloads[symbols]
benchmark = downloads[benchmark]['Close']



class MACrossoverStrategy(Strategy):
    buy_at_once_size = .01 # 1% #터틀에서 총자산대비 리스크로 사용하면 될듯

    def init(self, fast_period: int, slow_period: int):
        self.fast_ma = {}
        self.slow_ma = {}
        self.atr = {}

        for symbol in self.symbols:
            self.fast_ma[symbol] = talib.EMA(self.data[(symbol,'Close')], timeperiod=fast_period)
            self.slow_ma[symbol] = talib.EMA(self.data[(symbol,'Close')], timeperiod=slow_period)
            self.atr[symbol] = talib.ATR(self.data[(symbol,'High')], self.data[(symbol,'Low')], self.data[(symbol,'Close')], timeperiod=20)

    def next(self, i, record):
      # i는 시간이다. 계속 돌면서 시간이 갱신되는 것
      # record는 현재 시간까지 데이터??
      for symbol in self.symbols:
        if self.fast_ma[symbol][i-1] > self.slow_ma[symbol][i-1]:
            self.open(symbol=symbol, price=record[(symbol,'Open')], size=self.positionSize(record[(symbol,'Open')], self.atr[symbol][i]), stopLoss = record[(symbol,'Open')]-self.atr[symbol][i])


      for position in self.open_positions[:]: # [:]리스트 맨 앞부터 맨 뒤까지 불러오는 슬라이싱, i가 돌면서 매 순간마다 포지션을 수시로 불러오는 것다. 그리고 조건이 되면 close를 하는 것이고
        # 매 순간마다 포지션을 모두 불러오고 해당 순간의 가격에 대해서 조건이 맞으면 close를 하는 로직인데.. 처음 진입 시점의 atr을 알아야된다고...
        # record[(position.symbol,'Open')]은 i시점에서 해당 심볼의 OHLC값을 모두 가지고 있는 것이다!

        # if self.fast_ma[position.symbol][i-1] < self.slow_ma[position.symbol][i-1]:
        #   self.close(position=position, price=record[(position.symbol,'Open')])
        if position.stopLoss > record[(position.symbol,'Open')]:
          self.close(position=position, price=record[(position.symbol, 'Open')])

    #진입한 atr기준으로 손절값을 잡는데, 실제 손절은 어떻게 구현하지...?


    def positionSize(self, price: float, atr: float):
        return round((self.cash+self.assets_value) * self.buy_at_once_size / (2*atr)) if price > 0 else 0


backtest = Backtest(MACrossoverStrategy, data, commission=.001, cash=1e6)
result = backtest.run(20,60)

qs.reports.metrics(result.returns, benchmark)

