import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc
import argparse
from datetime import datetime, timedelta
import openai
import os
from dotenv import load_dotenv
    
class StockAnalyzer:
    """주식 데이터 분석 및 거래 전략 백테스트 클래스"""
    # 그리드 서치 파라미터를 한 곳에서 관리
    GRID_SEARCH_PARAMS = {
        'sma_crossover': {
'short_window': [3, 5, 7, 10],
'long_window': [10, 15, 20, 30]
},
        'macd': {
'fast': [5, 8, 12],
'slow': [10, 17, 26],
'signal': [5, 9, 12]
},
        'rsi': {
'window': [7, 10, 14, 20],
'buy_th': [30, 35, 40, 45],
'sell_th': [55, 60, 65, 70]
},
        'obv': {
'obv_window': [3, 5, 7, 10]
}
    }
    STRATEGY_PARAMS = {
        'sma_crossover': {'short_window': 3, 'long_window': 15},
        'macd': {'fast': 8, 'slow': 17, 'signal': 12},
        'rsi': {'window': 14, 'buy_th': 45, 'sell_th': 65},
        'obv': {'obv_window': 10}
    }

    def __init__(self, ticker, start_date, end_date, initial_capital=100_000_000):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data = None
        self.strategies = {
            'sma_crossover': self._sma_crossover_strategy,
            'macd': self._macd_strategy,
            'rsi': self._rsi_strategy,
            'obv': self._obv_strategy,
            'combined': self._combined_strategy
        }

    def fetch_data(self, verbose=False):
        """주식 데이터를 yfinance로 다운로드"""
        if verbose:
            print(f"{self.ticker} 데이터 다운로드 중...")
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date, auto_adjust=False)
        if self.data.empty:
            raise ValueError(f"{self.ticker}에 대한 데이터를 가져올 수 없습니다.")
        if verbose:
            print(f"데이터 {len(self.data)}건 다운로드 완료\n{self.data.head()}")
        return self.data

    # def calculate_indicators(self):
    #         """기술적 지표 계산"""
    #         self.data['SMA_5'] = self.data['Close'].rolling(window=5).mean()
    #         self.data['SMA_10'] = self.data['Close'].rolling(window=10).mean()
    #         self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
    #         self.data['EMA_5'] = self.data['Close'].ewm(span=5, adjust=False).mean()
    #         self.data['EMA_10'] = self.data['Close'].ewm(span=10, adjust=False).mean()
    #         self.data['MACD'] = self.data['EMA_10'] - self.data['EMA_5']
    #         self.data['Signal_Line'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
    #         self.data['MACD_Histogram'] = self.data['MACD'] - self.data['Signal_Line']
    #         delta = self.data['Close'].diff()
    #         gain = delta.where(delta > 0, 0)
    #         loss = -delta.where(delta < 0, 0)
    #         avg_gain = gain.rolling(window=14).mean()
    #         avg_loss = loss.rolling(window=14).mean()
    #         rs = avg_gain / avg_loss
    #         self.data['RSI'] = 100 - (100 / (1 + rs))
    #         self.data['Volume_SMA_5'] = self.data['Volume'].rolling(window=5).mean()
    #         self.data['OBV'] = 0
    #         self.data['OBV'] = np.where(
    #             self.data['Close'] > self.data['Close'].shift(1),
    #             self.data['Volume'],
    #             np.where(
    #                 self.data['Close'] < self.data['Close'].shift(1),
    #                 -self.data['Volume'],
    #                 0
    #             )
    #         )
    #         self.data['OBV'] = self.data['OBV'].cumsum()
    #         self.data = self.data.dropna()
    #         return self.data

    def _sma_crossover_strategy(self, data):
        params = self.STRATEGY_PARAMS['sma_crossover']
        short = params['short_window']
        long = params['long_window']

        data['SMA_short'] = data['Close'].rolling(window=short).mean()
        data['SMA_long'] = data['Close'].rolling(window=long).mean()
        data['Signal'] = 0
        data.loc[(data['SMA_short'] > data['SMA_long']) & (data['SMA_short'].shift(1) <= data['SMA_long'].shift(1)), 'Signal'] = 1
        data.loc[(data['SMA_short'] < data['SMA_long']) & (data['SMA_short'].shift(1) >= data['SMA_long'].shift(1)), 'Signal'] = -1
        data['Position'] = data['Signal'].replace(to_replace=0, value=np.nan).ffill().fillna(0)
        return data

    def _macd_strategy(self, data):
        params = self.STRATEGY_PARAMS['macd']
        fast = params['fast']
        slow = params['slow']
        signal = params['signal']

        data['EMA_fast'] = data['Close'].ewm(span=fast, adjust=False).mean()
        data['EMA_slow'] = data['Close'].ewm(span=slow, adjust=False).mean()
        data['MACD'] = data['EMA_fast'] - data['EMA_slow']
        data['Signal_Line'] = data['MACD'].ewm(span=signal, adjust=False).mean()
        data['Signal'] = 0
        data.loc[(data['MACD'] > data['Signal_Line']) & (data['MACD'].shift(1) <= data['Signal_Line'].shift(1)), 'Signal'] = 1
        data.loc[(data['MACD'] < data['Signal_Line']) & (data['MACD'].shift(1) >= data['Signal_Line'].shift(1)), 'Signal'] = -1
        data['Position'] = data['Signal'].replace(to_replace=0, value=np.nan).ffill().fillna(0)
        return data

    def _rsi_strategy(self, data):
        params = self.STRATEGY_PARAMS['rsi']
        window = params['window']
        buy_th = params['buy_th']
        sell_th = params['sell_th']

        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        data['Signal'] = 0
        data.loc[(data['RSI'] > buy_th) & (data['RSI'].shift(1) <= buy_th), 'Signal'] = 1
        data.loc[(data['RSI'] < sell_th) & (data['RSI'].shift(1) >= sell_th), 'Signal'] = -1
        data['Position'] = data['Signal'].replace(to_replace=0, value=np.nan).ffill().fillna(0)
        return data

    def _obv_strategy(self, data):
        params = self.STRATEGY_PARAMS['obv']
        obv_window = params['obv_window']

        data['OBV'] = 0
        data['OBV'] = np.where(data['Close'] > data['Close'].shift(1), data['Volume'], np.where(data['Close'] < data['Close'].shift(1), -data['Volume'], 0))
        data['OBV'] = data['OBV'].cumsum()
        data['OBV_SMA'] = data['OBV'].rolling(window=obv_window).mean()
        data['Signal'] = 0
        data.loc[(data['OBV'] > data['OBV_SMA']) & (data['OBV'].shift(1) <= data['OBV_SMA'].shift(1)), 'Signal'] = 1
        data.loc[(data['OBV'] < data['OBV_SMA']) & (data['OBV'].shift(1) >= data['OBV_SMA'].shift(1)), 'Signal'] = -1
        data['Position'] = data['Signal'].replace(to_replace=0, value=np.nan).ffill().fillna(0)
        return data

    def _combined_strategy(self, data, strategy_params=None):
        # strategy_params에서 각 전략별 파라미터 추출
        if strategy_params is None:
            strategy_params = self.STRATEGY_PARAMS
        # SMA
        sma_short = strategy_params['sma_crossover']['short_window']
        sma_long = strategy_params['sma_crossover']['long_window']
        data['SMA_short'] = data['Close'].rolling(window=sma_short).mean()
        data['SMA_long'] = data['Close'].rolling(window=sma_long).mean()
        data['SMA_Signal'] = 0
        data.loc[(data['SMA_short'] > data['SMA_long']) & (data['SMA_short'].shift(1) <= data['SMA_long'].shift(1)), 'SMA_Signal'] = 1
        data.loc[(data['SMA_short'] < data['SMA_long']) & (data['SMA_short'].shift(1) >= data['SMA_long'].shift(1)), 'SMA_Signal'] = -1
        # MACD
        macd_fast = strategy_params['macd']['fast']
        macd_slow = strategy_params['macd']['slow']
        macd_signal = strategy_params['macd']['signal']
        data['EMA_fast'] = data['Close'].ewm(span=macd_fast, adjust=False).mean()
        data['EMA_slow'] = data['Close'].ewm(span=macd_slow, adjust=False).mean()
        data['MACD'] = data['EMA_fast'] - data['EMA_slow']
        data['Signal_Line'] = data['MACD'].ewm(span=macd_signal, adjust=False).mean()
        data['MACD_Signal'] = 0
        data.loc[(data['MACD'] > data['Signal_Line']) & (data['MACD'].shift(1) <= data['Signal_Line'].shift(1)), 'MACD_Signal'] = 1
        data.loc[(data['MACD'] < data['Signal_Line']) & (data['MACD'].shift(1) >= data['Signal_Line'].shift(1)), 'MACD_Signal'] = -1
        # RSI
        rsi_window = strategy_params['rsi']['window']
        rsi_buy = strategy_params['rsi']['buy_th']
        rsi_sell = strategy_params['rsi']['sell_th']
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_window).mean()
        avg_loss = loss.rolling(window=rsi_window).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        data['RSI_Signal'] = 0
        data.loc[(data['RSI'] > rsi_buy) & (data['RSI'].shift(1) <= rsi_buy), 'RSI_Signal'] = 1
        data.loc[(data['RSI'] < rsi_sell) & (data['RSI'].shift(1) >= rsi_sell), 'RSI_Signal'] = -1
        # OBV
        obv_window = strategy_params['obv']['obv_window']
        data['OBV'] = np.where(data['Close'] > data['Close'].shift(1), data['Volume'],np.where(data['Close'] < data['Close'].shift(1), -data['Volume'], 0))
        data['OBV'] = data['OBV'].cumsum()
        data['OBV_SMA'] = data['OBV'].rolling(window=obv_window).mean()
        data['OBV_Signal'] = 0
        data.loc[(data['OBV'] > data['OBV_SMA']) & (data['OBV'].shift(1) <= data['OBV_SMA'].shift(1)), 'OBV_Signal'] = 1
        data.loc[(data['OBV'] < data['OBV_SMA']) & (data['OBV'].shift(1) >= data['OBV_SMA'].shift(1)), 'OBV_Signal'] = -1
        # 신호 집계
        data['Signal'] = 0
        buy_count = (data['SMA_Signal'].clip(lower=0) + data['MACD_Signal'].clip(lower=0) + data['RSI_Signal'].clip(lower=0) + data['OBV_Signal'].clip(lower=0))
        sell_count = (-data['SMA_Signal'].clip(upper=0) - data['MACD_Signal'].clip(upper=0) - data['RSI_Signal'].clip(upper=0) - data['OBV_Signal'].clip(upper=0))
        buy_signals = (buy_count >= 3)
        sell_signals = (sell_count >= 3)
        data.loc[buy_signals, 'Signal'] = 1
        data.loc[sell_signals, 'Signal'] = -1
        data['Position'] = data['Signal'].replace(to_replace=0, value=np.nan).ffill().fillna(0)
        return data

    def backtest(self, strategy_name='sma_crossover'):
        if self.data is None or self.data.empty:
            raise ValueError(f"{self.ticker}에 대한 데이터가 비어있습니다.")
        data = self.data.copy()
        strategy_func = self.strategies[strategy_name]
        data = strategy_func(data)
        data['Returns'] = data['Close'].pct_change()
        data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']
        data['Strategy_Returns'] = data['Strategy_Returns'].fillna(0)
        data['Cumulative_Returns'] = (1 + data['Returns']).cumprod()
        data['Strategy_Cumulative_Returns'] = (1 + data['Strategy_Returns']).cumprod()
        data['Portfolio_Value'] = self.initial_capital * data['Strategy_Cumulative_Returns']
        data['Trades'] = data['Signal'].abs()
        total_trades = data['Trades'].sum()
        total_return = data['Strategy_Cumulative_Returns'].iloc[-1] - 1
        buy_hold_return = data['Cumulative_Returns'].iloc[-1] - 1
        days = (data.index[-1] - data.index[0]).days
        annual_return = ((1 + total_return) ** (365 / days)) - 1 if days > 0 else 0
        rolling_max = data['Strategy_Cumulative_Returns'].cummax()
        drawdown = (data['Strategy_Cumulative_Returns'] / rolling_max) - 1
        max_drawdown = drawdown.min()
        print(f"\n===== {strategy_name} 전략 백테스팅 결과 =====")
        print(f"총 거래 횟수: {total_trades}")
        print(f"전략 총 수익률: {total_return:.2%}")
        print(f"매수 후 보유 수익률: {buy_hold_return:.2%}")
        print(f"연간 수익률: {annual_return:.2%}")
        print(f"최대 낙폭: {max_drawdown:.2%}")
        print(f"최종 포트폴리오 가치: {data['Portfolio_Value'].iloc[-1]:,.0f}원")
        self.data = data
        return {
            'total_trades': total_trades,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'final_value': data['Portfolio_Value'].iloc[-1]
        }

    def plot_results(self, strategy_name='sma_crossover', strategy_params=None):
        # strategy_params가 있으면 해당 파라미터로 지표 계산
        if strategy_params is None:
            strategy_params = self.STRATEGY_PARAMS
        data = self.data.copy()
        # SMA
        sma_short = strategy_params['sma_crossover']['short_window']
        sma_long = strategy_params['sma_crossover']['long_window']
        data['SMA_5'] = data['Close'].rolling(window=sma_short).mean()
        data['SMA_10'] = data['Close'].rolling(window=sma_long).mean()
        # MACD
        macd_fast = strategy_params['macd']['fast']
        macd_slow = strategy_params['macd']['slow']
        macd_signal = strategy_params['macd']['signal']
        data['EMA_fast'] = data['Close'].ewm(span=macd_fast, adjust=False).mean()
        data['EMA_slow'] = data['Close'].ewm(span=macd_slow, adjust=False).mean()
        data['MACD'] = data['EMA_fast'] - data['EMA_slow']
        data['Signal_Line'] = data['MACD'].ewm(span=macd_signal, adjust=False).mean()
        data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
        # RSI
        rsi_window = strategy_params['rsi']['window']
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_window).mean()
        avg_loss = loss.rolling(window=rsi_window).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        # OBV
        obv_window = strategy_params['obv']['obv_window']
        data['OBV'] = np.where(data['Close'] > data['Close'].shift(1), data['Volume'],
                              np.where(data['Close'] < data['Close'].shift(1), -data['Volume'], 0))
        data['OBV'] = data['OBV'].cumsum()
        data['OBV_SMA_5'] = data['OBV'].rolling(window=obv_window).mean()
        self.data = data
        rc('font', family='Malgun Gothic')
        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
        ax1 = plt.subplot(gs[0])
        ax1.plot(self.data.index, self.data['Close'], label='종가', alpha=0.7)
        ax1.plot(self.data.index, self.data['SMA_5'], label='5일 이동평균', alpha=0.7)
        ax1.plot(self.data.index, self.data['SMA_10'], label='10일 이동평균', alpha=0.7)
        buy_signals = self.data[self.data['Signal'] == 1]
        sell_signals = self.data[self.data['Signal'] == -1]
        ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', s=100, label='매수 신호')
        ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', s=100, label='매도 신호')
        ax1.set_title(f'{self.ticker} - {strategy_name} 전략 백테스팅 결과', fontsize=15)
        ax1.set_ylabel('가격', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True)
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(self.data.index, self.data['Cumulative_Returns'], label='매수 후 보유', alpha=0.7)
        ax2.plot(self.data.index, self.data['Strategy_Cumulative_Returns'], label='전략', alpha=0.7)
        ax2.set_ylabel('누적 수익률', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True)
        ax3 = plt.subplot(gs[2], sharex=ax1)
        if strategy_name == 'macd':
            ax3.plot(self.data.index, self.data['MACD'], label='MACD', alpha=0.7)
            ax3.plot(self.data.index, self.data['Signal_Line'], label='시그널 라인', alpha=0.7)
            ax3.bar(self.data.index, self.data['MACD_Histogram'], label='MACD 히스토그램', alpha=0.5)
            ax3.set_ylabel('MACD', fontsize=12)
        elif strategy_name == 'rsi':
            ax3.plot(self.data.index, self.data['RSI'], label='RSI', alpha=0.7)
            ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax3.set_ylabel('RSI', fontsize=12)
        elif strategy_name == 'obv':
            ax3.plot(self.data.index, self.data['OBV'], label='OBV', alpha=0.7)
            if 'OBV_SMA_5' in self.data.columns:
                ax3.plot(self.data.index, self.data['OBV_SMA_5'], label='OBV 5일 이동평균', alpha=0.7)
            ax3.set_ylabel('OBV', fontsize=12)
        else:
            ax3.plot(self.data.index, self.data['SMA_5'] - self.data['SMA_10'], label='이동평균선 차이', alpha=0.7)
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax3.set_ylabel('이동평균선 차이', fontsize=12)
        ax3.legend(loc='best')
        ax3.grid(True)
        ax4 = plt.subplot(gs[3], sharex=ax1)
        ax4.plot(self.data.index, self.data['Portfolio_Value'], label='포트폴리오 가치', alpha=0.7)
        ax4.set_ylabel('포트폴리오 가치 (원)', fontsize=12)
        ax4.set_xlabel('날짜', fontsize=12)
        ax4.legend(loc='best')
        ax4.grid(True)
        plt.tight_layout()
        plt.show()

    def compare_strategies(self):
        results = {}
        rc('font', family='Malgun Gothic')
        comparison_data = {}
        for strategy_name in self.strategies.keys():
            print(f"\n{strategy_name} 전략 백테스팅 중...")
            analyzer = StockAnalyzer(self.ticker, self.start_date, self.end_date, self.initial_capital)
            analyzer.fetch_data(verbose=False)
            results[strategy_name] = analyzer.backtest(strategy_name)
            comparison_data[strategy_name] = analyzer.data['Strategy_Cumulative_Returns']
        comparison = pd.DataFrame(results).T
        comparison.columns = ['총 거래 횟수', '총 수익률', '매수 후 보유 수익률', '연간 수익률', '최대 낙폭', '최종 포트폴리오 가치']
        comparison['총 수익률'] = comparison['총 수익률'].apply(lambda x: f"{x:.2%}")
        comparison['매수 후 보유 수익률'] = comparison['매수 후 보유 수익률'].apply(lambda x: f"{x:.2%}")
        comparison['연간 수익률'] = comparison['연간 수익률'].apply(lambda x: f"{x:.2%}")
        comparison['최대 낙폭'] = comparison['최대 낙폭'].apply(lambda x: f"{x:.2%}")
        comparison['최종 포트폴리오 가치'] = comparison['최종 포트폴리오 가치'].apply(lambda x: f"{x:,.0f}원")
        print("\n===== 전략 비교 =====")
        print(comparison)
        buy_hold_cumulative = (1 + self.data['Close'].pct_change()).cumprod()
        plt.figure(figsize=(12, 6))
        for strategy_name, strategy_cumulative_returns in comparison_data.items():
            plt.plot(self.data.index, strategy_cumulative_returns, label=strategy_name)
        plt.plot(self.data.index, buy_hold_cumulative, label='매수 후 보유', linestyle='--')
        plt.title('전략별 누적 수익률 비교', fontsize=15)
        plt.xlabel('날짜', fontsize=12)
        plt.ylabel('누적 수익률', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return comparison

    def grid_search(self, strategy_name):
        best_result = None
        best_params = None
        results = []
        params = self.GRID_SEARCH_PARAMS.get(strategy_name)
        if strategy_name == 'sma_crossover' and params:
            for short in params['short_window']:
                for long in params['long_window']:
                    if short >= long:
                        continue
                    data = self.data.copy()
                    data['SMA_short'] = data['Close'].rolling(window=short).mean()
                    data['SMA_long'] = data['Close'].rolling(window=long).mean()
                    data['Signal'] = 0
                    data.loc[(data['SMA_short'] > data['SMA_long']) & (data['SMA_short'].shift(1) <= data['SMA_long'].shift(1)), 'Signal'] = 1
                    data.loc[(data['SMA_short'] < data['SMA_long']) & (data['SMA_short'].shift(1) >= data['SMA_long'].shift(1)), 'Signal'] = -1
                    data['Position'] = data['Signal'].replace(to_replace=0, value=np.nan).ffill().fillna(0)
                    data['Returns'] = data['Close'].pct_change()
                    data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']
                    data['Strategy_Returns'] = data['Strategy_Returns'].fillna(0)
                    data['Strategy_Cumulative_Returns'] = (1 + data['Strategy_Returns']).cumprod()
                    total_return = data['Strategy_Cumulative_Returns'].iloc[-1] - 1
                    results.append({'params': (short, long), 'total_return': total_return})
                    print(f"short_window={short}, long_window={long} -> 총 수익률: {total_return:.2%}")
                    if best_result is None or total_return > best_result:
                        best_result = total_return
                        best_params = (short, long)
            print(f"[sma_crossover] 최적 파라미터: short_window={best_params[0]}, long_window={best_params[1]}, 총 수익률: {best_result:.2%}")
            return best_params, best_result
        elif strategy_name == 'macd' and params:
            for fast in params['fast']:
                for slow in params['slow']:
                    if fast >= slow:
                        continue
                    for signal in params['signal']:
                        data = self.data.copy()
                        data['EMA_fast'] = data['Close'].ewm(span=fast, adjust=False).mean()
                        data['EMA_slow'] = data['Close'].ewm(span=slow, adjust=False).mean()
                        data['MACD'] = data['EMA_fast'] - data['EMA_slow']
                        data['Signal_Line'] = data['MACD'].ewm(span=signal, adjust=False).mean()
                        data['Signal'] = 0
                        data.loc[(data['MACD'] > data['Signal_Line']) & (data['MACD'].shift(1) <= data['Signal_Line'].shift(1)), 'Signal'] = 1
                        data.loc[(data['MACD'] < data['Signal_Line']) & (data['MACD'].shift(1) >= data['Signal_Line'].shift(1)), 'Signal'] = -1
                        data['Position'] = data['Signal'].replace(to_replace=0, value=np.nan).ffill().fillna(0)
                        data['Returns'] = data['Close'].pct_change()
                        data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']
                        data['Strategy_Returns'] = data['Strategy_Returns'].fillna(0)
                        data['Strategy_Cumulative_Returns'] = (1 + data['Strategy_Returns']).cumprod()
                        total_return = data['Strategy_Cumulative_Returns'].iloc[-1] - 1
                        results.append({'params': (fast, slow, signal), 'total_return': total_return})
                        print(f"fast={fast}, slow={slow}, signal={signal} -> 총 수익률: {total_return:.2%}")
                        if best_result is None or total_return > best_result:
                            best_result = total_return
                            best_params = (fast, slow, signal)
            print(f"[macd] 최적 파라미터: fast={best_params[0]}, slow={best_params[1]}, signal={best_params[2]}, 총 수익률: {best_result:.2%}")
            return best_params, best_result
        elif strategy_name == 'rsi' and params:
            for window in params['window']:
                for buy_th in params['buy_th']:
                    for sell_th in params['sell_th']:
                        if buy_th >= sell_th:
                            continue
                        data = self.data.copy()
                        delta = data['Close'].diff()
                        gain = delta.where(delta > 0, 0)
                        loss = -delta.where(delta < 0, 0)
                        avg_gain = gain.rolling(window=window).mean()
                        avg_loss = loss.rolling(window=window).mean()
                        rs = avg_gain / avg_loss
                        data['RSI'] = 100 - (100 / (1 + rs))
                        data['Signal'] = 0
                        data.loc[(data['RSI'] > buy_th) & (data['RSI'].shift(1) <= buy_th), 'Signal'] = 1
                        data.loc[(data['RSI'] < sell_th) & (data['RSI'].shift(1) >= sell_th), 'Signal'] = -1
                        data['Position'] = data['Signal'].replace(to_replace=0, value=np.nan).ffill().fillna(0)
                        data['Returns'] = data['Close'].pct_change()
                        data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']
                        data['Strategy_Returns'] = data['Strategy_Returns'].fillna(0)
                        data['Strategy_Cumulative_Returns'] = (1 + data['Strategy_Returns']).cumprod()
                        total_return = data['Strategy_Cumulative_Returns'].iloc[-1] - 1
                        results.append({'params': (window, buy_th, sell_th), 'total_return': total_return})
                        print(f"window={window}, buy_th={buy_th}, sell_th={sell_th} -> 총 수익률: {total_return:.2%}")
                        if best_result is None or total_return > best_result:
                            best_result = total_return
                            best_params = (window, buy_th, sell_th)
            print(f"[rsi] 최적 파라미터: window={best_params[0]}, buy_th={best_params[1]}, sell_th={best_params[2]}, 총 수익률: {best_result:.2%}")
            return best_params, best_result
        elif strategy_name == 'obv' and params:
            for obv_window in params['obv_window']:
                data = self.data.copy()
                data['OBV'] = 0
                data['OBV'] = np.where(
                    data['Close'] > data['Close'].shift(1),
                    data['Volume'],
                    np.where(
                        data['Close'] < data['Close'].shift(1),
                        -data['Volume'],
                        0
                    )
                )
                data['OBV'] = data['OBV'].cumsum()
                data['OBV_SMA'] = data['OBV'].rolling(window=obv_window).mean()
                data['Signal'] = 0
                data.loc[(data['OBV'] > data['OBV_SMA']) & (data['OBV'].shift(1) <= data['OBV_SMA'].shift(1)), 'Signal'] = 1
                data.loc[(data['OBV'] < data['OBV_SMA']) & (data['OBV'].shift(1) >= data['OBV_SMA'].shift(1)), 'Signal'] = -1
                data['Position'] = data['Signal'].replace(to_replace=0, value=np.nan).ffill().fillna(0)
                data['Returns'] = data['Close'].pct_change()
                data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']
                data['Strategy_Returns'] = data['Strategy_Returns'].fillna(0)
                data['Strategy_Cumulative_Returns'] = (1 + data['Strategy_Returns']).cumprod()
                total_return = data['Strategy_Cumulative_Returns'].iloc[-1] - 1
                results.append({'params': (obv_window,), 'total_return': total_return})
                print(f"obv_window={obv_window} -> 총 수익률: {total_return:.2%}")
                if best_result is None or total_return > best_result:
                    best_result = total_return
                    best_params = (obv_window,)
            print(f"[obv] 최적 파라미터: obv_window={best_params[0]}, 총 수익률: {best_result:.2%}")
            return best_params, best_result
        else:
            print("해당 전략은 그리드 서치가 지원되지 않습니다.")
            return None, None

def llm_thinking(analyzer, strategy_name='sma_crossover', max_rows=30, bt_result=None):
    """
    OpenAI LLM을 활용해 주식 전략 결과를 요약하고, 전문가 관점의 투자의견/전략추천/리포트를 생성합니다.
    """
    from fear_and_greed import get as get_fng
    import yfinance as yf
    import datetime
    load_dotenv(override=True)
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("[경고] OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")
        return None
    openai.api_key = openai_api_key

    # 탐욕공포지수 데이터 추가
    try:
        fng_data = get_fng()
        fng_text = f"[탐욕공포지수] value: {fng_data.value}, description: {fng_data.description}, last_update: {fng_data.last_update.date()}\n"
    except Exception as e:
        fng_text = f"[탐욕공포지수를 가져오는 데 실패했습니다: {e}]\n"

    # VIX 데이터 추가
    try:
        vix_ticker = yf.Ticker("^VIX")
        today = datetime.date.today().strftime('%Y-%m-%d')
        yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        vix_data = vix_ticker.history(start=yesterday, end=today)
        today_vix = vix_data['Close']
        today_vix.reset_index(drop=True, inplace=True)
        vix_value = today_vix[0].round(2)
        vix_text = f"[VIX 변동성지수] value: {vix_value} (기준일: {today})\n"
    except Exception as e:
        vix_text = f"[VIX 변동성지수를 가져오는 데 실패했습니다: {e}]\n"

    # 최근 max_rows 데이터 요약
    data = analyzer.data.copy().tail(max_rows)
    summary = []
    for idx, row in data.iterrows():
        close_val = row['Close']
        if isinstance(close_val, pd.Series):
            close_val = close_val.iloc[0]
        signal_val = row['Signal'] if 'Signal' in row else 0
        if isinstance(signal_val, pd.Series):
            signal_val = signal_val.iloc[0]
        position_val = row['Position'] if 'Position' in row else 0
        if isinstance(position_val, pd.Series):
            position_val = position_val.iloc[0]
        summary.append(
            f"날짜: {idx.date()} | 종가: {float(close_val):.2f} | 매수/매도 신호: {int(signal_val)} | 전략 포지션: {int(position_val)}"
        )

    summary_text = '\n'.join(summary)
    # 전략별 백테스트 주요 결과
    if bt_result is None:
        bt_result = analyzer.backtest(strategy_name)
    result_text = (
        f"총 거래 횟수: {bt_result['total_trades']}\n"
        f"전략 총 수익률: {bt_result['total_return']:.2%}\n"
        f"매수 후 보유 수익률: {bt_result['buy_hold_return']:.2%}\n"
        f"연간 수익률: {bt_result['annual_return']:.2%}\n"
        f"최대 낙폭: {bt_result['max_drawdown']:.2%}\n"
        f"최종 포트폴리오 가치: {bt_result['final_value']:,.0f}원"
    )
    # 프롬프트 설계 (주식 트레이딩 전문가 관점)
    prompt = f"""
        너는 주식 트레이딩 전문가다. 아래는 {analyzer.ticker} 종목의 최근 전략 결과와 데이터 요약이다.\n\n        {fng_text}{vix_text}
        [최근 데이터 요약]
        {summary_text}

        [전략별 백테스트 결과]
        {result_text}

        아래 3가지를 전문가 관점에서 한국어로 자연스럽게 작성해줘.
        1. 현재 전략에 대한 투자의견 (매수/매도/관망 등)
        2. 전략 추천 및 이유
        3. 데이터와 전략 결과를 바탕으로 한 리포트 (시장 상황, 리스크, 참고사항 등)

        각 항목을 번호로 구분해서 5~10문장 이내로 구체적으로 작성해줘."""
    # OpenAI API 호출
    try:
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7,
            stream=False
        )
        result = completion.choices[0].message.content
        print(f"\n===== LLM 전문가 의견 =====\n{result}")
    except Exception as e:
        print(f"[OpenAI API 오류] {e}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description='주식 거래 결정 시스템')
    parser.add_argument('--ticker', type=str, default='AAPL', help='주식 티커 심볼 (기본값: AAPL, Apple Inc.)')
    parser.add_argument('--start_date', type=str, default=(datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d'), help='시작 날짜 (기본값: 5년 전)')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='종료 날짜 (기본값: 오늘)')
    parser.add_argument('--strategy', type=str, default='sma_crossover', choices=['sma_crossover', 'macd', 'rsi', 'obv', 'combined'], help='거래 전략 (기본값: sma_crossover)')
    parser.add_argument('--capital', type=float, default=100000000, help='초기 자본금 (기본값: 100,000,000원)')
    parser.add_argument('--compare', action='store_true', help='모든 전략을 비교합니다')
    parser.add_argument('--grid_search', type=str, metavar='STRATEGY', help='전략별 파라미터 그리드 서치를 수행합니다 (예: --grid_search macd)')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"\n===== 주식 거래 결정 시스템 =====")
    print(f"티커: {args.ticker}")
    print(f"기간: {args.start_date} ~ {args.end_date}")
    print(f"전략: {args.strategy}")
    print(f"초기 자본금: {args.capital:,.0f}원")
    analyzer = StockAnalyzer(args.ticker, args.start_date, args.end_date, args.capital)
    analyzer.fetch_data()
    if args.grid_search:
        analyzer.grid_search(args.grid_search)
    elif args.compare:
        analyzer.compare_strategies()
    else:
        bt_result = analyzer.backtest(args.strategy)
        analyzer.plot_results(args.strategy)
        llm_thinking(analyzer, args.strategy, bt_result=bt_result)

if __name__ == "__main__":
    main()