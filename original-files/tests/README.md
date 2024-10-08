# btester -   Multi-Assets Backtesting Framework

[![Test btester](https://github.com/pawelkn/btester/actions/workflows/test-btester.yml/badge.svg)](https://github.com/pawelkn/btester/actions/workflows/test-btester.yml) [![PyPi](https://img.shields.io/pypi/v/btester.svg)](https://pypi.python.org/pypi/btester/) [![Downloads](https://img.shields.io/pypi/dm/btester)](https://pypi.python.org/pypi/btester/) [![Codecov](https://codecov.io/gh/pawelkn/btester/branch/master/graph/badge.svg)](https://codecov.io/gh/pawelkn/btester/)

`btester` is a Python framework optimized for running backtests on multiple asset portfolios.

It provides tools for backtesting trading strategies based on historical market data. The framework includes classes for managing financial positions, completed trades, and a flexible abstract base class for implementing custom trading strategies.

## Installation

You can install `btester` using pip. Simply run the following command:

```bash
pip install myBtester
```

## Usage

1. Define your custom trading strategy by creating a class that inherits from the `Strategy` abstract class.

2. Implement the required methods in your custom strategy: `init` for initialization and `next` for the core strategy logic.

3. Instantiate the `Backtest` class with your custom strategy, historical market data, and other parameters.

4. Run the backtest using the `run` method, which returns a `Result` object containing backtest results.

## Example Usage

```python
# Example usage of the myBtester
from myBtester import Strategy, Backtest
import pandas as pd


# Define a custom strategy by inheriting from the abstract Strategy class
class MyStrategy(Strategy):
    def init(self):
        # Custom initialization logic for the strategy
        pass

    def next(self, i: int, record: Dict[Hashable, Any]):
        # Custom strategy logic for each time step
        pass


# Load historical market data
data = pd.read_csv('historical_data.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Initialize and run the backtest
backtest = Backtest(strategy=MyStrategy, data=data, cash=10000, commission=0.01)
result = backtest.run()

# Access backtest results
returns_series = result.returns
completed_trades = result.trades
remaining_positions = result.open_positions
```

## Examples

Check out the examples in the `examples` directory for additional use cases and demonstrations. The examples cover various scenarios and strategies to help you understand the versatility of the `btester`.

- [Example 1: Multi-Assets Moving Average Crossover Strategy](https://colab.research.google.com/github/pawelkn/btester/blob/master/examples/multi-assets-ma-crossover.ipynb)
- [Example 2: Multi-Assets Breakout Strategy](https://colab.research.google.com/github/pawelkn/btester/blob/master/examples/multi-assets-brakeout.ipynb)
- [Example 3: Single Asset Moving Average Crossover Strategy](https://colab.research.google.com/github/pawelkn/btester/blob/master/examples/single-asset-ma-crossover.ipynb)
- [Example 4: Single Asset Breakout Strategy](https://colab.research.google.com/github/pawelkn/btester/blob/master/examples/single-asset-brakeout.ipynb)

Feel free to explore and adapt these examples to suit your specific needs and trading strategies.
