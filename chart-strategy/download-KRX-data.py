import FinanceDataReader as fdr
import pandas as pd
import numpy as np


def _get_data(tickers, start):
    combined_df = pd.DataFrame()
    colIndex = []
    for code in tickers:
        df = fdr.DataReader(code, start)
        df.drop(labels=['Change'], inplace=True, axis=1)
        df.ffill(inplace=True)  # 결측값을 앞으로 채움  data.fillna(method='ffill', limit=1)
        # 여기까제 데이터를 가져오면 아래에서 데이터를 합치는 일을 함 마지막에는 멀티 인덱스로 변환암
        combined_df = df if combined_df.empty else pd.concat([combined_df, df], axis=1)
        # 이거 순서 중요하다..
        colIndex.append(("Open", code))
        colIndex.append(("High", code))
        colIndex.append(("Low", code))
        colIndex.append(("Close", code))
        colIndex.append(("Volume", code))
        print(df)

    col = pd.MultiIndex.from_tuples(colIndex, names=['Price', 'Ticker'])
    combined_df.columns = col

    # 이게 마지막 데이터프레임임 이걸 그대로 터틀코드에 사용이 가능함.
    return combined_df.swaplevel(axis=1)  # 멀티인덱스 데이터프레임에서 인덱스 또는 열의 레벨을 교환하는 메서드

kospi = fdr.StockListing('KOSPI')[['Code', 'Name']]  # 한국거래소: 7,632 종목
kosdaq = fdr.StockListing('KOSDAQ')[['Code', 'Name']]  # 한국거래소: 7,632 종목
syms = pd.concat([kospi, kosdaq])
# tickers = list(np.random.choice(syms['Code'].values, size=4000))  # 심볼들 중에서 랜덤하게 뽑는 것

tickers = ['000040', '263720', '166480']

mt_index = _get_data(kosdaq['Code'], '1990-03-03')  #'Code' 인덱스를 'Symbol'를 바꾸기.
# mt_index = _get_data(tickers, '2010-03-03')