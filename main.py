import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
from datetime import datetime, timedelta
from matplotlib import rc  # 추가

class StockAnalyzer:
    """주식 데이터 분석 및 거래 결정을 위한 클래스"""
    
    def __init__(self, ticker, start_date, end_date, initial_capital=100000000):
        """
        초기화 함수
        
        Args:
            ticker (str): 주식 티커 심볼
            start_date (str): 시작 날짜 (YYYY-MM-DD)
            end_date (str): 종료 날짜 (YYYY-MM-DD)
            initial_capital (float): 초기 자본금
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data = None
        self.strategies = {
            'sma_crossover': self._sma_crossover_strategy,
            'macd': self._macd_strategy,
            'rsi': self._rsi_strategy,
            'combined': self._combined_strategy,
            'obv': self._obv_strategy  # OBV 전략 추가
        }
        
    def fetch_data(self):
        """주식 데이터"""
        print(f"{self.ticker} 주식 데이터를 가져오는 중...")
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        
        if self.data.empty:
            raise ValueError(f"{self.ticker}에 대한 데이터를 가져올 수 없습니다.")
            
        print(f"데이터 가져오기 완료: {len(self.data)} 거래일")
        return self.data
    
    def calculate_indicators(self):
        """기술적 지표를 계산"""
        # 이동평균선
        self.data['SMA_5'] = self.data['Close'].rolling(window=5).mean() # 5일 이동평균선
        self.data['SMA_10'] = self.data['Close'].rolling(window=10).mean() # 10일 이동평균선
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean() # 20일 이동평균선
        
        # MACD (Moving Average Convergence Divergence)
        self.data['EMA_5'] = self.data['Close'].ewm(span=5, adjust=False).mean() # 5일 지수이동평균선
        self.data['EMA_10'] = self.data['Close'].ewm(span=10, adjust=False).mean() # 10일 지수이동평균선
        self.data['MACD'] = self.data['EMA_10'] - self.data['EMA_5']
        self.data['Signal_Line'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['Signal_Line']
        
        # RSI (Relative Strength Index)
        delta = self.data['Close'].diff() # 종가 변화량
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        self.data['RSI'] = 100 - (100 / (1 + rs)) # 14일 RSI, 30 이하 매수, 70 이상 매도
        
        # 거래량 지표
        self.data['Volume_SMA_5'] = self.data['Volume'].rolling(window=5).mean()
        
        # OBV 계산
        self.data['OBV'] = 0
        self.data['OBV'] = np.where(
            self.data['Close'] > self.data['Close'].shift(1),
            self.data['Volume'],
            np.where(
                self.data['Close'] < self.data['Close'].shift(1),
                -self.data['Volume'],
                0
            )
        )
        self.data['OBV'] = self.data['OBV'].cumsum()
        
        # NaN 값 제거
        self.data = self.data.dropna()
        
        return self.data
    
    def _sma_crossover_strategy(self, data):
        """이동평균선 교차 전략"""
        data['Signal'] = 0
        
        # 골든 크로스 (단기 이평선이 장기 이평선을 상향 돌파) 
        data.loc[(data['SMA_5'] > data['SMA_10']) & 
                 (data['SMA_5'].shift(1) <= data['SMA_10'].shift(1)), 'Signal'] = 1 
        
        # 데드 크로스 (단기 이평선이 장기 이평선을 하향 돌파)
        data.loc[(data['SMA_5'] < data['SMA_10']) & 
                 (data['SMA_5'].shift(1) >= data['SMA_10'].shift(1)), 'Signal'] = -1
        
        # 포지션 유지
        data['Position'] = data['Signal'].copy()
        data['Position'] = data['Position'].replace(to_replace=0, value=np.nan).ffill().fillna(0)
        
        return data
    
    def _macd_strategy(self, data):
        """MACD 전략"""
        data['Signal'] = 0
        
        # MACD가 시그널 라인을 상향 돌파 (매수 신호)
        data.loc[(data['MACD'] > data['Signal_Line']) & 
                 (data['MACD'].shift(1) <= data['Signal_Line'].shift(1)), 'Signal'] = 1
        
        # MACD가 시그널 라인을 하향 돌파 (매도 신호)
        data.loc[(data['MACD'] < data['Signal_Line']) & 
                 (data['MACD'].shift(1) >= data['Signal_Line'].shift(1)), 'Signal'] = -1
        
        # 포지션 유지
        data['Position'] = data['Signal'].copy()
        data['Position'] = data['Position'].replace(to_replace=0, value=np.nan).ffill().fillna(0)
        
        return data
    
    def _rsi_strategy(self, data):
        """RSI 전략"""
        data['Signal'] = 0
        
        # RSI가 40 이하에서 상승 (매수 신호) - 안전한 전략
        data.loc[(data['RSI'] > 40) & (data['RSI'].shift(1) <= 40), 'Signal'] = 1 
        
        # RSI가 60 이하에서 하락 (매도 신호) - 안전한 전략
        data.loc[(data['RSI'] < 60) & (data['RSI'].shift(1) >= 60), 'Signal'] = -1
        
        # 포지션 유지
        data['Position'] = data['Signal'].copy()
        data['Position'] = data['Position'].replace(to_replace=0, value=np.nan).ffill().fillna(0)
        
        return data
    
    def _obv_strategy(self, data):
        """OBV(온-밸런스 볼륨) 전략"""
        data['Signal'] = 0
        # OBV가 5일 OBV 이동평균선을 상향 돌파하면 매수, 하향 돌파하면 매도
        data['OBV_SMA_5'] = data['OBV'].rolling(window=5).mean()
        data.loc[
            (data['OBV'] > data['OBV_SMA_5']) &
            (data['OBV'].shift(1) <= data['OBV_SMA_5'].shift(1)),
            'Signal'
        ] = 1
        data.loc[
            (data['OBV'] < data['OBV_SMA_5']) &
            (data['OBV'].shift(1) >= data['OBV_SMA_5'].shift(1)),
            'Signal'
        ] = -1
        data['Position'] = data['Signal'].copy()
        data['Position'] = data['Position'].replace(to_replace=0, value=np.nan).ffill().fillna(0)
        return data
    
    def _combined_strategy(self, data):
        """여러 전략을 결합한 전략 (OBV 포함)"""
        # 각 전략의 신호를 계산
        data['Signal'] = 0

        # SMA 전략
        sma_data = data.copy()
        sma_data.loc[(sma_data['SMA_5'] > sma_data['SMA_10']) & 
                     (sma_data['SMA_5'].shift(1) <= sma_data['SMA_10'].shift(1)), 'Signal'] = 1
        sma_data.loc[(sma_data['SMA_5'] < sma_data['SMA_10']) & 
                     (sma_data['SMA_5'].shift(1) >= sma_data['SMA_10'].shift(1)), 'Signal'] = -1
        data['SMA_Signal'] = sma_data['Signal']

        # MACD 전략
        macd_data = data.copy()
        macd_data.loc[(macd_data['MACD'] > macd_data['Signal_Line']) & 
                      (macd_data['MACD'].shift() <= macd_data['Signal_Line'].shift()), 'Signal'] = 1
        macd_data.loc[(macd_data['MACD'] < macd_data['Signal_Line']) & 
                      (macd_data['MACD'].shift() >= macd_data['Signal_Line'].shift()), 'Signal'] = -1
        data['MACD_Signal'] = macd_data['Signal']

        # RSI 전략
        rsi_data = data.copy()
        rsi_data.loc[(rsi_data['RSI'] > 40) & (rsi_data['RSI'].shift(1) <= 40), 'Signal'] = 1
        rsi_data.loc[(rsi_data['RSI'] < 60) & (rsi_data['RSI'].shift(1) >= 60), 'Signal'] = -1
        data['RSI_Signal'] = rsi_data['Signal']

        # OBV 전략
        obv_data = data.copy()
        obv_data['OBV_SMA_5'] = obv_data['OBV'].rolling(window=5).mean()
        obv_data['OBV_Signal'] = 0
        obv_data.loc[(obv_data['OBV'] > obv_data['OBV_SMA_5']) & (obv_data['OBV'].shift(1) <= obv_data['OBV_SMA_5'].shift(1)), 'OBV_Signal'] = 1
        obv_data.loc[(obv_data['OBV'] < obv_data['OBV_SMA_5']) & (obv_data['OBV'].shift(1) >= obv_data['OBV_SMA_5'].shift(1)), 'OBV_Signal'] = -1
        data['OBV_Signal'] = obv_data['OBV_Signal']

        # 결합된 신호 계산 (4개 전략 중 3개 이상 동의 시 신호)
        data['Signal'] = 0
        buy_count = (data['SMA_Signal'].clip(lower=0) +
                     data['MACD_Signal'].clip(lower=0) +
                     data['RSI_Signal'].clip(lower=0) +
                     data['OBV_Signal'].clip(lower=0))
        sell_count = (-data['SMA_Signal'].clip(upper=0) -
                      data['MACD_Signal'].clip(upper=0) -
                      data['RSI_Signal'].clip(upper=0) -
                      data['OBV_Signal'].clip(upper=0))
        buy_signals = (buy_count >= 3)
        sell_signals = (sell_count >= 3)
        data.loc[buy_signals, 'Signal'] = 1
        data.loc[sell_signals, 'Signal'] = -1

        # 포지션 유지
        data['Position'] = data['Signal'].copy()
        data['Position'] = data['Position'].replace(to_replace=0, value=np.nan).ffill().fillna(0)
        return data

    def apply_strategy(self, strategy_name='sma_crossover'):
        """선택한 전략을 적용합니다."""
        if strategy_name not in self.strategies:
            raise ValueError(f"지원하지 않는 전략입니다: {strategy_name}")
        
        strategy_func = self.strategies[strategy_name]
        return strategy_func(self.data)
    
    def backtest(self, strategy_name='sma_crossover'):
        if self.data is None:
            self.fetch_data()
            self.calculate_indicators()
        data = self.data.copy()
        # 전략 함수가 복사본을 받아서 결과 반환
        strategy_func = self.strategies[strategy_name]
        data = strategy_func(data)
        
        # 수익률 계산
        data['Returns'] = data['Close'].pct_change()
        data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']
        
        # 누적 수익률 계산
        data['Cumulative_Returns'] = (1 + data['Returns']).cumprod()
        data['Strategy_Cumulative_Returns'] = (1 + data['Strategy_Returns']).cumprod()
        
        # 포트폴리오 가치 계산
        data['Portfolio_Value'] = self.initial_capital * data['Strategy_Cumulative_Returns']
        
        # 거래 횟수 계산
        data['Trades'] = data['Signal'].abs()
        total_trades = data['Trades'].sum()
        
        # 성과 지표 계산
        total_return = data['Strategy_Cumulative_Returns'].iloc[-1] - 1
        buy_hold_return = data['Cumulative_Returns'].iloc[-1] - 1
        
        # 연간 수익률 계산
        days = (data.index[-1] - data.index[0]).days
        annual_return = ((1 + total_return) ** (365 / days)) - 1 if days > 0 else 0
        
        # 최대 낙폭 계산
        rolling_max = data['Strategy_Cumulative_Returns'].cummax()
        drawdown = (data['Strategy_Cumulative_Returns'] / rolling_max) - 1
        max_drawdown = drawdown.min()
        
        # 결과 출력
        print(f"\n===== {strategy_name} 전략 백테스팅 결과 =====")
        print(f"총 거래 횟수: {total_trades}")
        print(f"전략 총 수익률: {total_return:.2%}")
        print(f"매수 후 보유 수익률: {buy_hold_return:.2%}")
        print(f"연간 수익률: {annual_return:.2%}")
        print(f"최대 낙폭: {max_drawdown:.2%}")
        print(f"최종 포트폴리오 가치: {data['Portfolio_Value'].iloc[-1]:,.0f}원")
        
        return {
            'total_trades': total_trades,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'final_value': data['Portfolio_Value'].iloc[-1]
        }
    

    def plot_results(self, strategy_name='sma_crossover'):
        """백테스팅 결과를 시각화합니다."""
        if 'Strategy_Cumulative_Returns' not in self.data.columns:
            self.backtest(strategy_name)
        
        # 한글 폰트 설정
        rc('font', family='Malgun Gothic')  # Windows의 '맑은 고딕' 폰트 사용
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
        
        # 주가 및 이동평균선 차트
        ax1 = plt.subplot(gs[0])
        ax1.plot(self.data.index, self.data['Close'], label='종가', alpha=0.7)
        ax1.plot(self.data.index, self.data['SMA_5'], label='5일 이동평균', alpha=0.7)
        ax1.plot(self.data.index, self.data['SMA_10'], label='10일 이동평균', alpha=0.7)
        
        # 매수/매도 신호 표시
        buy_signals = self.data[self.data['Signal'] == 1]
        sell_signals = self.data[self.data['Signal'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', s=100, label='매수 신호')
        ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', s=100, label='매도 신호')
        
        ax1.set_title(f'{self.ticker} - {strategy_name} 전략 백테스팅 결과', fontsize=15)
        ax1.set_ylabel('가격', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True)
        
        # 수익률 비교 차트
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(self.data.index, self.data['Cumulative_Returns'], label='매수 후 보유', alpha=0.7)
        ax2.plot(self.data.index, self.data['Strategy_Cumulative_Returns'], label='전략', alpha=0.7)
        ax2.set_ylabel('누적 수익률', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True)
        
        # 기술적 지표 차트 (전략에 따라 다른 지표 표시)
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
        
        # 포트폴리오 가치 차트
        ax4 = plt.subplot(gs[3], sharex=ax1)
        ax4.plot(self.data.index, self.data['Portfolio_Value'], label='포트폴리오 가치', alpha=0.7)
        ax4.set_ylabel('포트폴리오 가치 (원)', fontsize=12)
        ax4.set_xlabel('날짜', fontsize=12)
        ax4.legend(loc='best')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def compare_strategies(self):
        """여러 전략의 성능을 비교합니다."""
        results = {}
        rc('font', family='Malgun Gothic')  # Windows의 '맑은 고딕' 폰트 사용
        for strategy_name in self.strategies.keys():
            print(f"\n{strategy_name} 전략 백테스팅 중...")
            results[strategy_name] = self.backtest(strategy_name)
        
        # 결과 비교 테이블 생성
        comparison = pd.DataFrame(results).T
        comparison.columns = ['총 거래 횟수', '총 수익률', '매수 후 보유 수익률', '연간 수익률', '최대 낙폭', '최종 포트폴리오 가치']
        comparison['총 수익률'] = comparison['총 수익률'].apply(lambda x: f"{x:.2%}")
        comparison['매수 후 보유 수익률'] = comparison['매수 후 보유 수익률'].apply(lambda x: f"{x:.2%}")
        comparison['연간 수익률'] = comparison['연간 수익률'].apply(lambda x: f"{x:.2%}")
        comparison['최대 낙폭'] = comparison['최대 낙폭'].apply(lambda x: f"{x:.2%}")
        comparison['최종 포트폴리오 가치'] = comparison['최종 포트폴리오 가치'].apply(lambda x: f"{x:,.0f}원")
        
        print("\n===== 전략 비교 =====")
        print(comparison)
        
        # 누적 수익률 비교 차트
        buy_hold_cumulative = (1 + self.data['Close'].pct_change()).cumprod()
        plt.figure(figsize=(12, 6))
        for strategy_name in self.strategies.keys():
            self.apply_strategy(strategy_name)
            if 'Returns' not in self.data.columns:
                self.data['Returns'] = self.data['Close'].pct_change()
            self.data[f'{strategy_name}_returns'] = self.data['Position'].shift(1) * self.data['Returns']
            self.data[f'{strategy_name}_cumulative'] = (1 + self.data[f'{strategy_name}_returns']).cumprod()
            plt.plot(self.data.index, self.data[f'{strategy_name}_cumulative'], label=strategy_name)
        plt.plot(self.data.index, buy_hold_cumulative, label='매수 후 보유', linestyle='--')
        plt.title('전략별 누적 수익률 비교', fontsize=15)
        plt.xlabel('날짜', fontsize=12)
        plt.ylabel('누적 수익률', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return comparison

def parse_args():
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='주식 거래 결정 시스템')
    
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='주식 티커 심볼 (기본값: AAPL, Apple Inc.)')
    
    parser.add_argument('--start_date', type=str, default=(datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d'),
                        help='시작 날짜 (기본값: 5년 전)')
    
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                        help='종료 날짜 (기본값: 오늘)')
    
    parser.add_argument('--strategy', type=str, default='sma_crossover',
                        choices=['sma_crossover', 'macd', 'rsi', 'combined', 'obv'],
                        help='거래 전략 (기본값: sma_crossover)')
    
    parser.add_argument('--capital', type=float, default=100000000,
                        help='초기 자본금 (기본값: 100,000,000원)')
    
    parser.add_argument('--compare', action='store_true',
                        help='모든 전략을 비교합니다')
    
    return parser.parse_args()

def main():
    """메인 함수"""
    args = parse_args()
    
    print(f"\n===== 주식 거래 결정 시스템 =====")
    print(f"티커: {args.ticker}")
    print(f"기간: {args.start_date} ~ {args.end_date}")
    print(f"전략: {args.strategy}")
    print(f"초기 자본금: {args.capital:,.0f}원")
    
    # 주식 분석기 초기화
    analyzer = StockAnalyzer(args.ticker, args.start_date, args.end_date, args.capital)
    
    # 데이터 가져오기 및 지표 계산
    analyzer.fetch_data()
    analyzer.calculate_indicators()
    
    if args.compare:
        # 모든 전략 비교
        analyzer.compare_strategies()
    else:
        # 선택한 전략 백테스팅 및 시각화
        analyzer.backtest(args.strategy)
        analyzer.plot_results(args.strategy)

if __name__ == "__main__":
    main()
