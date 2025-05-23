# 주식 거래 결정 시스템

주식 거래 결정 시스템은 다양한 기술적 지표 기반의 전략(이동평균선, MACD, RSI, OBV, 복합전략 등)으로 주식 데이터를 분석하고, 백테스트 및 전략 비교, 파라미터 최적화, LLM 기반 전문가 의견 리포트까지 제공하는 Python 기반의 백테스팅/분석 도구입니다.

## 주요 기능
- yfinance를 통한 주가 데이터 자동 다운로드
- SMA, MACD, RSI, OBV, 복합 전략 백테스트 및 시각화
- 전략별 파라미터 그리드서치(최적화)
- 여러 전략 성과 비교
- OpenAI LLM을 활용한 전문가 관점 리포트 자동 생성
- CLI 및 인터랙티브 모드 지원

## 설치 방법
1. Python 3.8 이상 설치
2. 필수 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```
   또는 직접 설치:
   ```bash
   pip install yfinance pandas numpy matplotlib openai python-dotenv
   ```
3. OpenAI API 키 발급 후 `.env` 파일에 추가
   ```env
   OPENAI_API_KEY=sk-xxxxxx
   ```

## 사용법
### 1. 명령행 실행
```bash
python main.py --ticker AAPL --start_date 2020-01-01 --end_date 2024-01-01 --strategy macd --capital 100000000
```
- `--compare` : 모든 전략 성과 비교
- `--grid_search STRATEGY` : 해당 전략의 파라미터 최적화

### 2. 인터랙티브 모드
```bash
python main.py
```
입력 프롬프트에 따라 티커, 기간, 전략, 자본금 등 직접 입력 가능

## 주요 옵션
- `--ticker` : 분석할 주식 티커 (기본값: AAPL)
- `--start_date` : 시작 날짜 (YYYY-MM-DD)
- `--end_date` : 종료 날짜 (YYYY-MM-DD)
- `--strategy` : sma_crossover, macd, rsi, obv, combined 중 선택
- `--capital` : 초기 자본금 (기본값: 100,000,000)
- `--compare` : 모든 전략 성과 비교
- `--grid_search STRATEGY` : 전략별 파라미터 그리드서치

## 예시
- MACD 전략 백테스트:
  ```bash
  python main.py --ticker TSLA --strategy macd
  ```
- RSI 전략 파라미터 최적화:
  ```bash
  python main.py --ticker MSFT --grid_search rsi
  ```
- 모든 전략 비교:
  ```bash
  python main.py --ticker AAPL --compare
  ```

## 참고
- LLM 리포트 기능은 OpenAI API 키 필요
- 일부 기능(탐욕공포지수 등)은 외부 API 연결 필요

## 라이선스
MIT License
