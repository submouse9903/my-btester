import backtrader as bt
import backtrader.analyzers as btanalyzers
import matplotlib
from datetime import datetime
import pandas as pd
import requests
import yfinance as yf


assets = 'nasdaq100'
benchmark = 'NQ=F'
ldf = pd.read_html(requests.get(f'https://www.slickcharts.com/{assets}', headers={'User-agent': 'Mozilla/5.0'}).text)
symbols = [x.replace('.','-') for x in ldf[0]['Symbol'] if isinstance(x, str)]
downloads = yf.download([*symbols, benchmark], '2000-01-01', '2024-07-01', group_by='ticker')
benchmark = downloads[benchmark]['Close']

data = downloads[symbols]


class MaCrossStrategy(bt.Strategy):
    params = (
        ('fast_length', 5),
        ('slow_length', 25)
    )

    def __init__(self):
        # print('__init__')
        self.crossovers = []

        for d in self.datas: #adddata에서 추가한 각 주식에 대한 반복문임
            ma_fast = bt.ind.SMA(d, period=self.params.fast_length)
            ma_slow = bt.ind.SMA(d, period=self.params.slow_length)

            self.crossovers.append(bt.ind.CrossOver(ma_fast, ma_slow))

    def next(self): #next를 할때마다 모든 주식에 대해서 처음부터 시작하는게 문제 같은데...
        # 근데 이건 btester랑 다르게 record값이 없어서...
        for i, d in enumerate(self.datas):
            # print(i, d)
            if not self.getposition(d).size:
                if self.crossovers[i] > 0:
                    self.buy(data=d, size=None, )
            elif self.crossovers[i] < 0:
                self.close(data=d)


import time

df_swp = data.swaplevel(0, 1, axis=1)



def do_backtest(data, number):
    cerebro = bt.Cerebro()

    for symbol in symbols:
        symbol_data = bt.feeds.PandasData(dataname=df_swp.xs(symbol, level=1, axis=1))
        cerebro.adddata(symbol_data)

    cerebro.addstrategy(MaCrossStrategy)

    cerebro.broker.setcash(100000.0)

    cerebro.addsizer(bt.sizers.PercentSizer, percents=10) #매매 사이즈 정하는 곳
    cerebro.broker.setcommission(commission=0.0025)

    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')

    back = cerebro.run()



if __name__ == "__main__":

    start_time = time.time()

    params = zip(os.listdir("data"), range(len(df_swp.columns.levels[0])))

    with Pool() as p:
        results = p.starmap(do_backtest, params)


    time_taken = time.time() - start_time
    print(f"Took {time_taken} seconds")

    print(results)



portfolio_stats = back[0].analyzers.getbyname('PyFolio')
returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
returns.index = returns.index.tz_convert(None)

import quantstats as qs
qs.reports.metrics(returns, benchmark)
qs.plots.log_returns(returns, benchmark, fontname='sans-serif')
qs.plots.drawdown(returns, figsize=(10,3), fontname='sans-serif')


