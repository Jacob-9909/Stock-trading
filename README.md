# 📊 주식 거래 결정 시스템 (Stock Trading Decision System)

Python 기반의 포괄적인 주식 백테스팅 및 분석 도구로, 다양한 기술적 지표를 활용한 거래 전략의 성과를 분석하고 AI 기반 투자 리포트를 제공합니다.

## ✨ 주요 기능

### 📈 지원하는 거래 전략
- **SMA Crossover**: 단기/장기 이동평균선 교차 전략
- **MACD**: 지수이동평균 수렴확산지수 전략
- **RSI**: 상대강도지수 기반 과매수/과매도 전략
- **Bollinger Bands**: 볼린저 밴드 기반 평균회귀 전략
- **OBV**: 거래량 기반 온밸런스볼륨 전략
- **Combined Strategy**: 5개 전략의 신호를 조합한 복합 전략

### 🔧 핵심 기능
- **yfinance 기반 데이터 수집**: 실시간 주가 데이터 자동 다운로드
- **백테스팅 엔진**: 실제 거래 시뮬레이션으로 전략 성과 측정
- **파라미터 최적화**: 그리드 서치를 통한 전략별 최적 파라미터 탐색
- **전략 성과 비교**: 여러 전략의 성과를 동시에 비교 분석
- **고급 시각화**: matplotlib을 활용한 다층 차트 분석
- **AI 리포트**: OpenAI GPT를 활용한 전문가 수준의 투자 분석 리포트
- **시장 지표 통합**: Fear & Greed Index, VIX 등 시장 심리 지표 포함

### 📊 분석 지표
- **수익성 지표**: 총 수익률, 연간 수익률, 매수후보유 대비 성과
- **위험 지표**: 최대 낙폭(Max Drawdown), 변동성 분석
- **거래 통계**: 총 거래 횟수, 승률, 평균 수익/손실
- **포트폴리오 가치**: 실시간 포트폴리오 가치 추적

## 🚀 설치 및 설정

### 1. 환경 요구사항
- Python 3.11 이상
- 필요한 패키지들이 requirements.txt에 정의됨

### 2. 패키지 설치
```bash
# requirements.txt를 이용한 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install yfinance pandas numpy matplotlib openai python-dotenv fear-and-greed
```

### 3. 환경 변수 설정
`.env` 파일을 생성하고 다음 API 키들을 설정하세요:
```env
# OpenAI API 키 (AI 리포트 기능용)
OPENAI_API_KEY=sk-your-openai-api-key

# Financial Modeling Prep API 키 (기업 정보용, 선택사항)
FMP_API_KEY=your-fmp-api-key
```

## 💻 사용법

### 1. 명령행 인터페이스 (CLI)
```bash
# 기본 백테스팅
python main.py --ticker AAPL --strategy macd

# 특정 기간 분석
python main.py --ticker TSLA --start_date 2020-01-01 --end_date 2024-01-01 --strategy rsi

# 초기 자본금 설정
python main.py --ticker MSFT --capital 50000000 --strategy bollinger

# 파라미터 최적화
python main.py --ticker NVDA --grid_search macd

# 모든 전략 성과 비교
python main.py --ticker GOOGL --compare
```

### 2. 인터랙티브 모드
```bash
# 대화형 모드로 실행
python main.py
```
프롬프트에 따라 티커, 기간, 전략, 자본금 등을 단계별로 입력할 수 있습니다.

## 📋 명령행 옵션

| 옵션 | 설명 | 기본값 | 예시 |
|------|------|--------|------|
| `--ticker` | 주식 티커 심볼 | AAPL | `--ticker TSLA` |
| `--start_date` | 분석 시작 날짜 | 5년 전 | `--start_date 2020-01-01` |
| `--end_date` | 분석 종료 날짜 | 오늘 | `--end_date 2024-12-31` |
| `--strategy` | 사용할 거래 전략 | sma_crossover | `--strategy macd` |
| `--capital` | 초기 투자 자본금(원) | 100,000,000 | `--capital 50000000` |
| `--compare` | 모든 전략 성과 비교 | - | `--compare` |
| `--grid_search` | 파라미터 최적화 실행 | - | `--grid_search rsi` |

### 지원하는 전략 목록
- `sma_crossover`: 이동평균선 교차 전략
- `macd`: MACD 전략  
- `rsi`: RSI 전략
- `bollinger`: 볼린저 밴드 전략
- `obv`: 온밸런스볼륨 전략
- `combined`: 복합 전략

## 📊 출력 결과

### 1. 백테스팅 결과
```
===== macd 전략 백테스팅 결과 =====
총 거래 횟수: 45
전략 총 수익률: 23.45%
매수 후 보유 수익률: 18.72%
연간 수익률: 8.91%
최대 낙폭: -12.34%
최종 포트폴리오 가치: 123,450,000원
```

### 2. 시각화 차트
- **가격 차트**: 주가와 매수/매도 신호점 표시
- **수익률 차트**: 전략 수익률 vs 매수후보유 수익률 비교
- **지표 차트**: 각 전략별 기술적 지표 시각화
- **포트폴리오 차트**: 시간에 따른 포트폴리오 가치 변화

### 3. AI 투자 리포트
OpenAI GPT를 활용하여 다음 내용을 포함한 전문가 수준의 분석 리포트를 생성합니다:
- 현 시점 투자 판단 및 근거
- 시장 상황 및 변동성 지표 해석
- 전략적 인사이트 및 추천
- 리스크 요인 분석
- 투자자 관점의 실질적 조언

## ⚙️ 전략별 파라미터

### SMA Crossover
- `short_window`: 단기 이동평균 기간 (기본: 3일)
- `long_window`: 장기 이동평균 기간 (기본: 15일)

### MACD
- `fast`: 빠른 EMA 기간 (기본: 8일)
- `slow`: 느린 EMA 기간 (기본: 17일)
- `signal`: 시그널 라인 기간 (기본: 12일)

### RSI
- `window`: RSI 계산 기간 (기본: 14일)
- `buy_th`: 매수 임계값 (기본: 45)
- `sell_th`: 매도 임계값 (기본: 65)

### Bollinger Bands
- `bol_window`: 볼린저 밴드 기간 (기본: 20일)

### OBV
- `obv_window`: OBV 이동평균 기간 (기본: 10일)

## 📈 사용 예시

### 예시 1: Tesla 주식 MACD 전략 분석
```bash
python main.py --ticker TSLA --strategy macd --start_date 2022-01-01 --capital 50000000
```

### 예시 2: Apple 주식 RSI 파라미터 최적화
```bash
python main.py --ticker AAPL --grid_search rsi
```

### 예시 3: Microsoft 주식 모든 전략 성과 비교
```bash
python main.py --ticker MSFT --compare --start_date 2021-01-01
```

### 예시 4: 인터랙티브 모드로 종합 분석
```bash
python main.py
# 이후 대화형 프롬프트에 따라 입력:
# - 티커: NVDA
# - 시작 날짜: 2020-01-01
# - 종료 날짜: 2024-01-01
# - 전략: combined
# - 초기 자본: 100000000
# - 그리드서치: y
```

## 🔧 고급 기능

### 1. 복합 전략 (Combined Strategy)
5개 전략의 신호를 종합하여 다음 규칙으로 거래 신호를 생성:
- **매수 신호**: 3개 이상의 전략에서 동시에 매수 신호 발생
- **매도 신호**: 3개 이상의 전략에서 동시에 매도 신호 발생

### 2. 그리드 서치 최적화
각 전략별로 미리 정의된 파라미터 범위 내에서 최적의 조합을 탐색:
- 모든 파라미터 조합에 대해 백테스팅 실행
- 총 수익률 기준으로 최적 파라미터 선택
- 자동으로 전략 파라미터 업데이트 (선택사항)

### 3. 시장 지표 통합
- **Fear & Greed Index**: 시장 심리 상태 분석
- **VIX**: 변동성 지수로 시장 불안 정도 측정
- **기업 기본정보**: Beta, 시가총액, 52주 변동폭 등

## ⚠️ 주의사항

### 필수 API 키
- **OpenAI API**: LLM 리포트 기능 사용시 필요
- **FMP API**: 기업 기본정보 조회시 필요 (선택사항)

### 데이터 제한사항
- yfinance API의 일일 요청 제한이 있을 수 있음
- 일부 외부 API는 인터넷 연결 필요
- 과거 데이터 기반 백테스팅이므로 미래 성과를 보장하지 않음

### 성능 고려사항
- 그리드 서치는 대량의 계산을 수행하므로 시간이 오래 걸릴 수 있음
- 메모리 사용량은 분석 기간과 데이터 양에 비례

## 📁 프로젝트 구조

```
Stock-trading/
├── main.py              # 메인 실행 파일
├── requirements.txt     # 패키지 의존성
├── pyproject.toml      # 프로젝트 설정
├── README.md           # 프로젝트 문서
├── .env                # 환경 변수 (사용자 생성)
├── .gitignore         # Git 제외 파일
└── .python-version    # Python 버전 명시
```

## 🤝 기여하기

1. 이 저장소를 Fork합니다
2. 새로운 기능 브랜치를 생성합니다 (`git checkout -b feature/AmazingFeature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 Push합니다 (`git push origin feature/AmazingFeature`)
5. Pull Request를 생성합니다

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🔗 관련 링크

- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Financial Modeling Prep API](https://financialmodelingprep.com/developer/docs)

---

**⚠️ 면책조항**: 이 도구는 교육 및 연구 목적으로 제작되었습니다. 실제 투자 결정에 사용하기 전에 충분한 검토가 필요하며, 투자 손실에 대한 책임은 사용자에게 있습니다.
