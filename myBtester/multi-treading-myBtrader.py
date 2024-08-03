import datetime
import pandas_ta as ta
import pandas as pd
import os
import time

from backtesting import Backtest
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import GOOG

from multiprocessing import Pool

#

print("Hello")

class RsiOscillator(Strategy):

    upper_bound = 70
    lower_bound = 30
    rsi_window = 14

    # Do as much initial computation as possible
    def init(self):
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), self.rsi_window)

    # Step through bars one by one
    # Note that multiple buys are a thing here
    def next(self):

        if crossover(self.rsi, self.upper_bound):
            self.position.close()

        elif crossover(self.lower_bound, self.rsi):
            self.buy()

def do_backtest(filename, number):
    # print(filename)
    # data = pd.read_csv(f"data/{filename}",
    #         names=[
    #             "OpenTime",
    #             "Open",
    #             "High",
    #             "Low",
    #             "Close",
    #             "Volume",
    #             "CloseTime",
    #             "QuoteVolume",
    #             "NumTrades",
    #             "TakerBuyBaseVol",
    #             "TakerBuyQuoteVol",
    #             "Unused",
    # ])
    #
    # data["OpenTime"] = pd.to_datetime(data["OpenTime"], unit="ms")
    # data.set_index("OpenTime", inplace=True)

    bt = Backtest(filename, RsiOscillator, cash=10_000_000, commission=.002)
    stats = bt.run()
    sym = filename.split("-")[0]
    return (sym, stats["Return [%]"])


if __name__ == "__main__":

    start_time = time.time()

    params = zip(GOOG, range(len(GOOG)))

    with Pool() as p:
        results = p.starmap(do_backtest, params)


    time_taken = time.time() - start_time
    print(f"Took {time_taken} seconds")

    print(results)


























