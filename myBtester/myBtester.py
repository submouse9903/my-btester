from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Type, Hashable, Optional
from math import nan, isnan
import pandas as pd

# short포지션 구축 가능한지 알아보기
# windows git test, asvㄴㅇㄹㄴㅇㄹ
@dataclass
class Position:
    """
    현재 포지션을 나타냅니다.

    속성:
    - symbol: Optional[str] - 금융 상품의 심볼.
    - open_date: Optional[datetime] - 포지션이 개방된 날짜.
    - last_date: Optional[datetime] - 포지션의 최신 업데이트 날짜.
    - open_price: float - 포지션을 개방한 가격.
    - last_price: float - 상품의 최신 시장 가격.
    - position_size: float - 포지션의 크기.
    - profit_loss: float - 포지션의 누적 이익 또는 손실.
    - change_pct: float - 포지션 개방 이후 가격의 백분율 변화.
    - current_value: float - 포지션의 현재 시장 가치.

    메소드:
    - update(last_date: datetime, last_price: float) - 최신 시장 데이터로 포지션을 업데이트합니다.
    """
    symbol: Optional[str] = None
    open_date: Optional[datetime] = None
    last_date: Optional[datetime] = None
    open_price: float = nan
    last_price: float = nan
    position_size: float = nan
    profit_loss: float = nan
    change_pct: float = nan
    current_value: float = nan
    long: bool = nan #포지션 객체에 롱숏 구분할 수 있게 만든다.

    # 주식 분할이나 액면 병합 문제 => adjuested price로 해결 가능
    takeProfit: float = nan
    stopLoss: float = nan

    def update(self, last_date: datetime, last_price: float):
        self.last_date = last_date
        self.last_price = last_price
        self.profit_loss = (self.last_price - self.open_price) * self.position_size
        self.change_pct = (self.last_price / self.open_price - 1) * 100
        self.current_value = self.open_price * self.position_size + self.profit_loss

@dataclass
class Trade:
    """
    완료된 금융 거래를 나타냅니다.

    속성:
    - symbol: Optional[str] - 금융 상품의 심볼.
    - open_date: Optional[datetime] - 거래가 개시된 날짜.
    - close_date: Optional[datetime] - 거래가 종료된 날짜.
    - open_price: float - 거래를 개시한 가격.
    - close_price: float - 거래를 종료한 가격.
    - position_size: float - 거래된 포지션의 크기.
    - profit_loss: float - 거래의 누적 이익 또는 손실.
    - change_pct: float - 거래 기간 동안 가격의 백분율 변화.
    - trade_commission: float - 거래에 대해 지불된 수수료.
    - cumulative_return: float - 거래 후 누적 수익.
    """
    symbol: Optional[str] = None
    open_date: Optional[datetime] = None
    close_date: Optional[datetime] = None
    open_price: float = nan
    close_price: float = nan
    position_size: float = nan
    profit_loss: float = nan
    change_pct: float = nan
    trade_commission: float = nan
    cumulative_return: float = nan

    long: bool = nan

@dataclass
class Result:
    """
    백테스트 결과를 담는 컨테이너 클래스입니다.

    속성:
    - returns: pd.Series - 누적 수익률의 시계열 데이터입니다.
    - trades: List[Trade] - 완료된 거래의 목록입니다.
    - open_positions: List[Position] - 남아있는 개방 포지션의 목록입니다.
    """
    returns: pd.Series
    trades: List[Trade]
    open_positions: List[Position]

class Strategy(ABC):
    """
    트레이딩 전략을 구현하기 위한 추상 기본 클래스입니다.

    메소드:
    - init(self) - 전략에 필요한 리소스를 초기화하기 위한 추상 메소드입니다.
    - next(self, i: int, record: Dict[Hashable, Any]) - 전략의 핵심 기능을 정의하는 추상 메소드입니다.

    속성:
    - data: pd.DataFrame - 과거 시장 데이터입니다.
    - date: Optional[datetime] - 백테스팅 중의 현재 날짜입니다.
    - cash: float - 거래 가능한 현금입니다.
    - commission: float - 거래에 대한 수수료율입니다.
    - symbols: List[str] - 시장 데이터의 심볼 목록입니다.
    - records: List[Dict[Hashable, Any]] - 시장 데이터를 나타내는 레코드의 목록입니다.
    - index: List[datetime] - 시장 데이터에 해당하는 날짜의 목록입니다.
    - returns: List[float] - 백테스팅 중 누적 수익률의 목록입니다.
    - trades: List[Trade] - 백테스팅 중 완료된 거래의 목록입니다.
    - open_positions: List[Position] - 백테스팅 중 남아있는 개방 포지션의 목록입니다.
    - cumulative_return: float - 전략의 누적 수익률입니다.
    - assets_value: float - 개방 포지션의 시장 가치입니다. 현금은 제외된 값임

    + 손절라인까지 총자산 인식 includedSL_total_value

    메소드:
    - open(self, price: float, size: Optional[float] = None, symbol: Optional[str] = None) -> bool
    - close(self, price: float, symbol: Optional[str] = None, position: Optional[Position] = None) -> bool
    """

    @abstractmethod
    def init(self):
        """
        전략에 필요한 리소스와 매개변수를 초기화하기 위한 추상 메소드입니다.

        이 메소드는 백테스트가 시작될 때 한 번 호출되어 트레이딩 전략에 필요한 모든 설정이나 구성을 수행합니다.
        이를 통해 전략은 변수를 초기화하거나, 매개변수를 설정하거나, 전략의 기능에 필요한 외부 데이터를 로드할 수 있습니다.

        Parameters:
         - *args: 초기화 중에 전달할 수 있는 추가 위치 매개변수입니다.
         - **kwargs: 초기화 중에 전달할 수 있는 추가 키워드 매개변수입니다.

        Example:
        ```python
        def init(self, buy_period: int, sell_period: int):
            self.buy_signal = {}
            self.sell_signal = {}

            for symbol in self.symbols:
                self.buy_signal[symbol] = UpBreakout(self.data[(symbol,'Close')], buy_period)
                self.sell_signal[symbol] = DownBreakout(self.data[(symbol,'Close')], sell_period)
        ```

        Note:
        전략을 초기화할 때 유연성과 사용자 정의를 허용하기 위해 init 메소드 내에서 예상되는 매개변수와 그들의 기본값을 정의하는 것이 좋습니다.
        """

    @abstractmethod
    def next(self, i: int, record: Dict[Hashable, Any]):
        """
         전략의 핵심 기능을 정의하는 추상 메소드입니다.

         이 메소드는 각 시간 단계마다 반복적으로 호출되며, 전략이 'record'로 표현된 현재 시장 데이터를 기반으로 결정을 내릴 수 있게 합니다.
         이는 신호 생성, 포지션 관리, 거래 결정 등의 트레이딩 전략의 핵심 로직을 정의합니다.

        Parameters:
        - i (int): 현재 시간 단계의 인덱스입니다.
        - record (Dict[Hashable, Any]): 현재 시간 단계에서의 시장 데이터를 나타내는 사전입니다. 키는 심볼을 포함할 수 있고, 값은 관련 시장 데이터(예: OHLC 가격)를 포함할 수 있습니다.

        Example:
        ```python
        def next(self, i, record):
            for symbol in self.symbols:
                if self.buy_signal[symbol][i-1]:
                    self.open(symbol=symbol, price=record[(symbol,'Open')], size=self.positionSize(record[(symbol,'Open')]))

            for position in self.open_positions[:]:
                if self.sell_signal[position.symbol][i-1]:
                    self.close(position=position, price=record[(position.symbol,'Open')])
        ```
        """

    def __init__(self):
        self.data = pd.DataFrame()
        self.date = None
        self.cash = .0
        self.commission = .0

        self.symbols: List[str] = []

        self.records: List[Dict[Hashable, Any]] = []
        self.index: List[datetime] = []

        self.returns: List[float] = []
        self.trades: List[Trade] = []
        self.open_positions: List[Position] = []

        self.cumulative_return = self.cash
        self.assets_value = .0

    def open(self, price: float, size: Optional[float] = None, symbol: Optional[str] = None, takeProfit: Optional[float] = None, stopLoss: Optional[float] = None):
        """
        지정된 매개변수에 따라 새로운 금융 포지션을 개설합니다.

        매개변수:
        - price: float - 포지션을 개설할 가격입니다.
        - size: Optional[float] - 포지션의 크기입니다. 제공되지 않은 경우, 사용 가능한 현금에 기반하여 계산됩니다.
        - symbol: Optional[str] - 금융 상품의 심볼입니다.

        반환값:
        - bool: 포지션이 성공적으로 개설되면 True, 그렇지 않으면 False입니다.

        이 메소드는 새로운 포지션을 개설하는 비용을 계산하고, 지정된 크기가 사용 가능한 현금을 고려하여 실행 가능한지 확인하며, 전략의 개방 포지션을 그에 따라 업데이트합니다.
        포지션이 성공적으로 개설되면 True를 반환하고, 그렇지 않으면 False를 반환합니다.
        """
        if isnan(price) or price <= 0 or (size is not None and (isnan(size) or size <= .0)):
            return False

        if size is None:
            size = self.cash / (price * (1 + self.commission))
            open_cost = self.cash
        else:
            open_cost = size * price * (1 + self.commission)

        if isnan(size) or size <= .0 or self.cash < open_cost:
            return False


        position = Position(symbol=symbol, open_date=self.date, open_price=price, position_size=size, takeProfit=takeProfit, stopLoss=stopLoss)
        position.update(last_date=self.date, last_price=price)

        self.assets_value += position.current_value
        self.cash -= open_cost

        self.open_positions.extend([position])
        return True

    def close(self, price: float, symbol: Optional[str] = None, position: Optional[Position] = None):
        """
        지정된 매개변수에 따라 기존의 금융 포지션을 청산합니다.

        매개변수:
        - price: float - 포지션을 청산할 가격입니다.
        - symbol: Optional[str] - 금융 상품의 심볼입니다.
        - position: Optional[Position] - 청산할 특정 포지션입니다. 제공되지 않은 경우, 해당 심볼에 대한 모든 포지션을 청산합니다.

        반환값:
        - bool: 포지션(들)이 성공적으로 청산되면 True, 그렇지 않으면 False입니다.

        이 메소드는 포지션을 청산하는 비용을 계산하고, 전략의 누적 수익률을 업데이트하며, 거래 세부 사항을 기록합니다. 특정 포지션이 제공되면, 그 포지션만이 청산됩니다.
        포지션이 지정되지 않은 경우, 지정된 심볼에 대한 모든 개방 포지션을 청산합니다. 포지션(들)이 성공적으로 청산되면 True를 반환하고, 그렇지 않으면 False를 반환합니다.
        """
        if isnan(price) or price <= 0:
            return False

        if position is None:
            for position in self.open_positions[:]:
                if position.symbol == symbol:
                    self.close(position=position, price=price)
        else:
            self.assets_value -= position.current_value
            position.update(last_date=self.date, last_price=price)

            trade_commission = (position.open_price + position.last_price) * position.position_size * self.commission
            self.cumulative_return += position.profit_loss - trade_commission

            trade = Trade(position.symbol, position.open_date, position.last_date, position.open_price,
                position.last_price, position.position_size, position.profit_loss, position.change_pct,
                trade_commission, self.cumulative_return)

            self.trades.extend([trade])
            self.open_positions.remove(position)

            close_cost = position.last_price * position.position_size * self.commission
            self.cash += position.current_value - close_cost

        return True

    def __eval(self, *args, **kwargs):
        self.cumulative_return = self.cash
        self.assets_value = .0

        self.init(*args, **kwargs)

        for i, record in enumerate(self.records):
            self.date = self.index[i]

            self.next(i, record)

            for position in self.open_positions:
                last_price = record[(position.symbol, 'Close')] if (position.symbol, 'Close') in record else record['Close']
                if last_price > 0:
                    position.update(last_date=self.date, last_price=last_price)

            self.assets_value = sum(position.current_value for position in self.open_positions)
            self.returns.append(self.cash + self.assets_value)

        return Result(
            returns=pd.Series(index=self.index, data=self.returns, dtype=float),
            trades=self.trades,
            open_positions=self.open_positions
        )

class Backtest:
    """
    주어진 전략을 사용하여 과거 시장 데이터에 대한 백테스트를 실행하는 클래스입니다.

    속성:
    - strategy: Type[Strategy] - 백테스트할 전략의 유형입니다.
    - data: pd.DataFrame - 과거 시장 데이터입니다.
    - cash: float - 거래에 사용할 수 있는 초기 현금입니다.
    - commission: float - 거래에 대한 수수료율입니다.

    메소드:
    - run(*args, **kwargs) - 백테스트를 실행하고 결과를 반환합니다.
    """
    def __init__(self,
                 strategy: Type[Strategy],
                 data: pd.DataFrame,
                 cash: float = 10_000,
                 commission: float = .0
                 ):

        self.strategy = strategy
        self.data = data
        self.cash = cash
        self.commission = commission

        columns = data.columns
        self.symbols = columns.get_level_values(0).unique().tolist() if isinstance(columns, pd.MultiIndex) else []

        self.records = data.to_dict('records')
        self.index = data.index.tolist()

    def run(self, *args, **kwargs):
        strategy = self.strategy()
        strategy.data = self.data
        strategy.cash = self.cash
        strategy.commission = self.commission

        strategy.symbols = self.symbols
        strategy.records = self.records
        strategy.index = self.index

        return strategy._Strategy__eval(*args, **kwargs)