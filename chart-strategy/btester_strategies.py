from myBtester import Strategy
import pandas as pd
import talib


class MACrossoverStrategy_RiskManagement_ATR(Strategy):
    def init(self, fast_period: int, slow_period: int):
        self.max_total_unit_risk = .01  # 거래당 1% 리스크 지는 것
        self.highpoint_of_account = 0  # 초기 현금으로 최대 계좌 초기화
        self.account_pole = self.cash  # 자산 최대값 기준로 리스크단위 증감 계산하기 위함 초기 계좌기준 10%씩 늘어나면 갱신하는 구조로할까

        self.fast_ma = {}  # next에서 가격데이터랑 같은 날짜로 반복문이 돌려진다.
        self.slow_ma = {}
        self.sma_200 = {}
        self.atr = {}
        df = pd.DataFrame()  # Empty DataFrame

        for symbol in self.symbols:
            self.fast_ma[symbol] = talib.EMA(self.data[(symbol, 'Close')], timeperiod=fast_period)
            self.slow_ma[symbol] = talib.EMA(self.data[(symbol, 'Close')], timeperiod=slow_period)
            self.sma_200[symbol] = talib.SMA(self.data[(symbol, 'Close')], timeperiod=200)
            self.atr[symbol] = talib.ATR(self.data[(symbol, 'High')], self.data[(symbol, 'Low')],self.data[(symbol, 'Close')], timeperiod=20)

    def next(self, i, record):  # 지표까지 모두 준비된 데이터에 대해서 모두 동일한 시점에서 시작하는 부분이다. 35,285.57%
        # i는 이동하는 캔들이다. 계속 캔들을 갱신하는 것이다, 지표는 앞전에 init함수로 미리 데이터를 만들어 놓고, 이 부분에서 하나씩 갱신하는 것.
        self.riskSize()

        for symbol in self.symbols:
            open_condition1 = self.fast_ma[symbol][i - 2] < self.slow_ma[symbol][i - 2]
            open_condition2 = self.fast_ma[symbol][i - 1] > self.slow_ma[symbol][i - 1]
            long_condition_200sma = self.fast_ma[symbol][i - 1] > self.sma_200[symbol][i - 1]
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


class MACrossoverStrategy_RiskManagement_ATR_covariance(Strategy):
    def init(self, fast_period: int, slow_period: int):
        self.max_total_unit_risk = .01  # 거래당 1% 리스크 지는 것
        self.highpoint_of_account = 0  # 초기 현금으로 최대 계좌 초기화
        self.account_pole = self.cash  # 자산 최대값 기준로 리스크단위 증감 계산하기 위함 초기 계좌기준 10%씩 늘어나면 갱신하는 구조로할까

        self.fast_ma = {}  # next에서 가격데이터랑 같은 날짜로 반복문이 돌려진다.
        self.slow_ma = {}
        self.sma_200 = {}
        self.atr = {}
        df = pd.DataFrame()  # Empty DataFrame

        self.previous_open_positions = [] #포지션의 변동을 감지해서 포지션 변동해야지만 상관관계 계산하는 로직 수행

        for symbol in self.symbols:
            self.fast_ma[symbol] = talib.EMA(self.data[(symbol, 'Close')], timeperiod=fast_period)
            self.slow_ma[symbol] = talib.EMA(self.data[(symbol, 'Close')], timeperiod=slow_period)
            self.sma_200[symbol] = talib.SMA(self.data[(symbol, 'Close')], timeperiod=200)
            self.atr[symbol] = talib.ATR(self.data[(symbol, 'High')], self.data[(symbol, 'Low')],self.data[(symbol, 'Close')], timeperiod=20)

    def next(self, i, record):  # 지표까지 모두 준비된 데이터에 대해서 모두 동일한 시점에서 시작하는 부분이다. 35,285.57%
        # i는 이동하는 캔들이다. 계속 캔들을 갱신하는 것이다, 지표는 앞전에 init함수로 미리 데이터를 만들어 놓고, 이 부분에서 하나씩 갱신하는 것.

        # 현재 가지고있는 포지션에 대해서 상관관계가 낮은 자산 위주로 사는 것이 목적
        symbols = [position.symbol for position in self.open_positions[:]]
        # print(symbols)
        for symbol in self.symbols:
            open_condition1 = self.fast_ma[symbol][i - 2] < self.slow_ma[symbol][i - 2]
            open_condition2 = self.fast_ma[symbol][i - 1] > self.slow_ma[symbol][i - 1]
            long_condition_200sma = self.fast_ma[symbol][i - 1] > self.sma_200[symbol][i - 1]

            if open_condition1 and open_condition2 and self.atr[symbol][i] > 0:  # 해당 종목에서 포지션이 있더라도 조전에 만족하면 그냥 진입함. 피라미딩같은 느낌
                if symbols != []:
                    symbols.append(symbol)
                    corr = self.data.loc[:, symbols].xs('Close', level=1, axis=1).pct_change().corr()
                    max_corr = corr[corr[symbol] != 1][symbol].max()
                    if max_corr > 0.65: continue #현재 심볼과 보유한 포지션의 현재시점의 상관계수가 0.65이상이면 진입하지 않는다.

                position_size = self.positionSize(record[(symbol, 'Open')], self.atr[symbol][i])
                stop_loss = record[(symbol, 'Open')] - self.atr[symbol][i]
                self.open(symbol=symbol, price=record[(symbol, 'Open')], size=position_size, stopLoss=stop_loss, long=True)

        for position in self.open_positions[:]:  # [:]리스트 맨 앞부터 맨 뒤까지 불러오는 슬라이싱, i가 돌면서 매 순간마다 포지션을 수시로 불러오는 것다. 그리고 조건이 되면 close를 하는 것이고
            # 매 순간마다 포지션을 모두 불러오고 해당 순간의 가격에 대해서 조건이 맞으면 close를 하는 로직인데.. 처음 진입 시점의 atr을 알아야된다고...
            close_condition1 = self.fast_ma[position.symbol][i - 1] < self.slow_ma[position.symbol][i - 1]
            stoploss_condition1 = position.stopLoss > record[(position.symbol, 'Open')]
            if close_condition1 or stoploss_condition1:  # atr만이 청산전략이 아니다. 다른 가격적인 조건이 청산 조건이 추가되어야함. 청산에서 atr은 정해진 리스크량으로 정의하기 위해서 사용된 것
                # self.previous_open_positions = self.open_positions[:] #이게 필요한가....?
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
