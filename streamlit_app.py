import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from main import StockAnalyzer, llm_thinking
import os

warnings.filterwarnings('ignore')

# Streamlit 페이지 설정
st.set_page_config(
    page_title="주식 자동매매 전략 분석",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바에서 설정 옵션들
st.sidebar.title("🎯 주식 분석 설정")

# 종목 입력
ticker = st.sidebar.text_input("종목 심볼 입력", value="AAPL", help="예: AAPL, GOOGL, TSLA")

# 기간 설정
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.sidebar.date_input(
        "시작일",
        value=datetime.now() - timedelta(days=1095),  # 3년 전
        help="분석 시작 날짜"
    )
with col2:
    end_date = st.sidebar.date_input(
        "종료일",
        value=datetime.now(),
        help="분석 종료 날짜"
    )

# 초기 투자금
initial_capital = st.sidebar.number_input(
    "초기 투자금 (원)",
    min_value=1000000,
    max_value=10000000000,
    value=100000000,
    step=10000000,
    help="백테스트를 위한 초기 투자금"
)

# 전략 선택
strategy_options = {
    'sma_crossover': 'SMA 교차 전략',
    'macd': 'MACD 전략', 
    'rsi': 'RSI 전략',
    'bollinger': '볼린저 밴드 전략',
    'obv': 'OBV 전략',
    'combined': '통합 전략'
}

selected_strategy = st.sidebar.selectbox(
    "거래 전략 선택",
    options=list(strategy_options.keys()),
    format_func=lambda x: strategy_options[x],
    help="백테스트할 거래 전략을 선택하세요"
)

# 그리드 서치 옵션
enable_grid_search = st.sidebar.checkbox(
    "파라미터 최적화 실행",
    value=False,
    help="선택한 전략의 최적 파라미터를 찾습니다 (시간이 걸릴 수 있습니다)"
)

# 메인 타이틀
st.title("📈 주식 자동매매 전략 분석 시스템")
st.markdown("---")

# 분석 실행 버튼
if st.button("🚀 분석 시작", type="primary"):
    
    # 로딩 표시
    with st.spinner('데이터를 분석하고 있습니다...'):
        try:
            # StockAnalyzer 인스턴스 생성
            analyzer = StockAnalyzer(
                ticker=ticker,
                start_date=str(start_date),
                end_date=str(end_date),
                initial_capital=initial_capital
            )
            
            # 데이터 다운로드
            analyzer.fetch_data(verbose=True)
            
            # 그리드 서치 실행 (선택된 경우)
            if enable_grid_search:
                st.info("💡 파라미터 최적화를 실행 중입니다...")
                best_params, best_result = analyzer.grid_search(selected_strategy)
                if best_params:
                    st.success(f"✅ 최적 파라미터 발견: {best_params}, 수익률: {best_result:.2%}")
            
            # 백테스팅 실행
            bt_result = analyzer.backtest(selected_strategy)
              # 결과 표시 - 색상 지표와 함께
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # 총 수익률 - 양수면 녹색, 음수면 빨간색
                total_return_color = "normal" if bt_result['total_return'] >= 0 else "inverse"
                st.metric(
                    "💰 총 수익률",
                    f"{bt_result['total_return']:.2%}",
                    f"vs 매수보유: {bt_result['total_return'] - bt_result['buy_hold_return']:.2%}",
                    delta_color=total_return_color
                )
            
            with col2:
                # 연간 수익률 - 양수면 녹색, 음수면 빨간색
                annual_return_color = "normal" if bt_result['annual_return'] >= 0 else "inverse"
                st.metric(
                    "📅 연간 수익률", 
                    f"{bt_result['annual_return']:.2%}",
                    delta=f"{bt_result['annual_return']:.2%}",
                    delta_color=annual_return_color
                )
            
            with col3:
                # 최대 낙폭 - 항상 빨간색 (낙폭이므로)
                st.metric(
                    "⬇️ 최대 낙폭",
                    f"{bt_result['max_drawdown']:.2%}",
                    delta=f"{bt_result['max_drawdown']:.2%}",
                    delta_color="inverse"
                )
            
            # 추가 지표들
            col4, col5, col6 = st.columns(3)
            
            with col4:
                # 총 거래 횟수 - 중성 색상
                st.metric(
                    "🔄 총 거래 횟수",
                    f"{bt_result['total_trades']}"
                )
            
            with col5:
                # 최종 포트폴리오 가치 - 초기 자본 대비 증감
                portfolio_delta = bt_result['final_value'] - initial_capital
                portfolio_color = "normal" if portfolio_delta >= 0 else "inverse"
                st.metric(
                    "💼 최종 포트폴리오 가치",
                    f"{bt_result['final_value']:,.0f}원",
                    delta=f"{portfolio_delta:,.0f}원",
                    delta_color=portfolio_color
                )
            
            with col6:
                # 매수 후 보유 수익률 - 양수면 녹색, 음수면 빨간색
                buy_hold_color = "normal" if bt_result['buy_hold_return'] >= 0 else "inverse"
                st.metric(
                    "📈 매수 후 보유 수익률",
                    f"{bt_result['buy_hold_return']:.2%}",
                    delta=f"{bt_result['buy_hold_return']:.2%}",
                    delta_color=buy_hold_color                )
            
            # 성과 요약 카드들 (컬러풀한 스타일)
            st.markdown("---")
            st.subheader("📊 성과 요약")
            
            # CSS 스타일 추가
            st.markdown("""
            <style>
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .metric-card-positive {
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            }
            .metric-card-negative {
                background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
            }
            .metric-card-neutral {
                background: linear-gradient(135deg, #3f51b1 0%, #5a55ae 100%);
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            }
            .metric-label {
                font-size: 14px;
                opacity: 0.9;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # 성과 카드들
            performance_cols = st.columns(3)
            
            with performance_cols[0]:
                total_return_class = "metric-card-positive" if bt_result['total_return'] >= 0 else "metric-card-negative"
                st.markdown(f"""
                <div class="metric-card {total_return_class}">
                    <div class="metric-label">💰 전략 총 수익률</div>
                    <div class="metric-value">{bt_result['total_return']:.2%}</div>
                    <div class="metric-label">vs 매수보유: {bt_result['total_return'] - bt_result['buy_hold_return']:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with performance_cols[1]:
                annual_return_class = "metric-card-positive" if bt_result['annual_return'] >= 0 else "metric-card-negative"
                st.markdown(f"""
                <div class="metric-card {annual_return_class}">
                    <div class="metric-label">📅 연간 수익률</div>
                    <div class="metric-value">{bt_result['annual_return']:.2%}</div>
                    <div class="metric-label">연율화된 수익률</div>
                </div>
                """, unsafe_allow_html=True)
            
            with performance_cols[2]:
                st.markdown(f"""
                <div class="metric-card metric-card-negative">
                    <div class="metric-label">⬇️ 최대 낙폭 (MDD)</div>
                    <div class="metric-value">{bt_result['max_drawdown']:.2%}</div>
                    <div class="metric-label">최대 손실 구간</div>
                </div>
                """, unsafe_allow_html=True)
            
            performance_cols2 = st.columns(3)
            
            with performance_cols2[0]:
                st.markdown(f"""
                <div class="metric-card metric-card-neutral">
                    <div class="metric-label">🔄 총 거래 횟수</div>
                    <div class="metric-value">{bt_result['total_trades']}</div>
                    <div class="metric-label">매수/매도 신호 횟수</div>
                </div>
                """, unsafe_allow_html=True)
            
            with performance_cols2[1]:
                portfolio_delta = bt_result['final_value'] - initial_capital
                portfolio_class = "metric-card-positive" if portfolio_delta >= 0 else "metric-card-negative"
                st.markdown(f"""
                <div class="metric-card {portfolio_class}">
                    <div class="metric-label">💼 최종 포트폴리오</div>
                    <div class="metric-value">{bt_result['final_value']:,.0f}원</div>
                    <div class="metric-label">변동: {portfolio_delta:,.0f}원</div>
                </div>
                """, unsafe_allow_html=True)
            
            with performance_cols2[2]:
                buy_hold_class = "metric-card-positive" if bt_result['buy_hold_return'] >= 0 else "metric-card-negative"
                st.markdown(f"""
                <div class="metric-card {buy_hold_class}">
                    <div class="metric-label">📈 매수후보유 수익률</div>
                    <div class="metric-value">{bt_result['buy_hold_return']:.2%}</div>
                    <div class="metric-label">벤치마크 수익률</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 차트 생성
            st.subheader("📊 백테스팅 결과 차트")
            
            data = analyzer.data.copy()
            
            # Plotly 서브플롯 생성
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('주가 & 매매 신호', '누적 수익률 비교', '기술적 지표', '포트폴리오 가치'),
                row_width=[0.3, 0.2, 0.2, 0.3]
            )
            
            # 1. 주가 차트와 매매 신호
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='종가',
                    line=dict(color='blue', width=1.5)
                ),
                row=1, col=1
            )
            
            # 이동평균선 (5일, 10일)
            data['SMA_5'] = data['Close'].rolling(window=5).mean()
            data['SMA_10'] = data['Close'].rolling(window=10).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_5'],
                    mode='lines',
                    name='5일 이평선',
                    line=dict(color='orange', width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_10'],
                    mode='lines',
                    name='10일 이평선',
                    line=dict(color='purple', width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # 매수/매도 신호
            buy_signals = data[data['Signal'] == 1]
            sell_signals = data[data['Signal'] == -1]
            
            if not buy_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['Close'],
                        mode='markers',
                        name='매수 신호',
                        marker=dict(symbol='triangle-up', size=10, color='green')
                    ),
                    row=1, col=1
                )
            
            if not sell_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['Close'],
                        mode='markers',
                        name='매도 신호',
                        marker=dict(symbol='triangle-down', size=10, color='red')
                    ),
                    row=1, col=1
                )
            
            # 2. 누적 수익률 비교
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Cumulative_Returns'],
                    mode='lines',
                    name='매수 후 보유',
                    line=dict(color='gray', width=1.5)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Strategy_Cumulative_Returns'],
                    mode='lines',
                    name='전략 수익률',
                    line=dict(color='red', width=2)
                ),
                row=2, col=1
            )
            
            # 3. 기술적 지표 (전략별로 다르게)
            if selected_strategy == 'macd' and 'MACD' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue')
                    ),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Signal_Line'],
                        mode='lines',
                        name='시그널 라인',
                        line=dict(color='red')
                    ),
                    row=3, col=1
                )
                # MACD 히스토그램
                macd_histogram = data['MACD'] - data['Signal_Line']
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=macd_histogram,
                        name='MACD 히스토그램',
                        opacity=0.5,
                        marker_color='green'
                    ),
                    row=3, col=1
                )
            
            elif selected_strategy == 'rsi' and 'RSI' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple')
                    ),
                    row=3, col=1
                )
                # RSI 기준선들
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            elif selected_strategy == 'bollinger' and 'Upper_band' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='종가',
                        line=dict(color='blue')
                    ),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Upper_band'],
                        mode='lines',
                        name='상단 밴드',
                        line=dict(color='red')
                    ),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Lower_band'],
                        mode='lines',
                        name='하단 밴드',
                        line=dict(color='green')
                    ),
                    row=3, col=1
                )
            
            elif selected_strategy == 'obv' and 'OBV' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['OBV'],
                        mode='lines',
                        name='OBV',
                        line=dict(color='brown')
                    ),
                    row=3, col=1
                )
                if 'OBV_SMA' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['OBV_SMA'],
                            mode='lines',
                            name='OBV 이동평균',
                            line=dict(color='orange')
                        ),
                        row=3, col=1
                    )
            
            else:  # SMA Crossover 또는 기본
                if 'SMA_short' in data.columns and 'SMA_long' in data.columns:
                    sma_diff = data['SMA_short'] - data['SMA_long']
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=sma_diff,
                            mode='lines',
                            name='이동평균선 차이',
                            line=dict(color='green')
                        ),
                        row=3, col=1
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="black", row=3, col=1)
            
            # 4. 포트폴리오 가치
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Portfolio_Value'],
                    mode='lines',
                    name='포트폴리오 가치',
                    line=dict(color='darkgreen', width=2)
                ),
                row=4, col=1
            )
            
            # 레이아웃 설정
            fig.update_layout(
                height=800,
                title_text=f"{ticker} - {strategy_options[selected_strategy]} 백테스팅 결과",
                showlegend=True,
                hovermode='x unified'
            )
            
            fig.update_xaxes(title_text="날짜", row=4, col=1)
            fig.update_yaxes(title_text="가격", row=1, col=1)
            fig.update_yaxes(title_text="누적 수익률", row=2, col=1)
            fig.update_yaxes(title_text="지표 값", row=3, col=1)
            fig.update_yaxes(title_text="포트폴리오 가치", row=4, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
              # LLM 분석 결과
            st.markdown("---")
            
            # AI 분석 헤더 디자인
            st.markdown("""
            <div style="
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                margin: 20px 0;
                box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            ">
                <h2 style="color: white; margin: 0; font-size: 2.2rem;">
                    🤖 AI 전문가 투자 분석
                </h2>
                <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 1.1rem;">
                    GPT-4 기반 전문가 수준의 종합 투자 리포트
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # LLM 분석 실행
            with st.spinner('🧠 AI가 심층 투자 분석을 생성하고 있습니다...'):
                
                # llm_thinking 함수 실행하여 결과 캡처
                import io
                import sys
                from contextlib import redirect_stdout
                
                # stdout 캡처를 위한 StringIO 객체
                captured_output = io.StringIO()
                
                # llm_thinking 실행하고 출력 캡처
                with redirect_stdout(captured_output):
                    llm_thinking(analyzer, selected_strategy, bt_result=bt_result)
                
                # 캡처된 출력 가져오기
                llm_output = captured_output.getvalue()
                  # LLM 분석 결과 표시
                if llm_output and "===== LLM 전문가 의견 =====" in llm_output:
                    # LLM 출력에서 전문가 의견 부분만 추출
                    llm_result = llm_output.split("===== LLM 전문가 의견 =====")[1].strip()
                    
                    # AI 분석 결과를 마크다운으로 표시
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                        border: 1px solid #e1e8ed;
                        border-radius: 15px;
                        padding: 10px 0;
                        margin: 20px 0;
                        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                        position: relative;
                    ">
                        <div style="
                            position: absolute;
                            top: -10px;
                            left: 30px;
                            background: linear-gradient(45deg, #667eea, #764ba2);
                            color: white;
                            padding: 8px 20px;
                            border-radius: 20px;
                            font-size: 0.9rem;
                            font-weight: bold;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                        ">
                            💡 AI 투자 리포트
                        </div>
                    </div>
                    """, unsafe_allow_html=True)                    # 마크다운 형식으로 리포트 내용 렌더링
                    st.markdown(f"""
                    <div style="
                        background: white;
                        border-radius: 10px;
                        padding: 30px;
                        margin: 10px 0 20px 0;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                        border: 1px solid #e8ecef;
                        color: #2c3e50;
                        line-height: 1.6;
                    ">
                    <style>
                        .ai-report h2 {{ color: #1f2937; font-weight: bold; margin-top: 20px; margin-bottom: 10px; }}
                        .ai-report h3 {{ color: #374151; font-weight: 600; margin-top: 15px; margin-bottom: 8px; }}
                        .ai-report strong {{ color: #111827; font-weight: 600; }}
                        .ai-report em {{ color: #4b5563; font-style: italic; }}
                        .ai-report ul, .ai-report ol {{ margin: 10px 0; padding-left: 20px; }}
                        .ai-report li {{ margin: 5px 0; color: #374151; }}
                    </style>
                    <div class="ai-report">
                    {llm_result}
                    </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 추가 정보 박스
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #ffeef8 0%, #f0e6ff 100%);
                        border-left: 5px solid #9c27b0;
                        padding: 20px;
                        margin: 20px 0;
                        border-radius: 0 15px 15px 0;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    ">
                        <h4 style="color: #7b1fa2; margin: 0 0 10px 0;">
                            ⚠️ 투자 유의사항
                        </h4>
                        <p style="color: #4a148c; margin: 0; line-height: 1.6;">
                            본 AI 분석은 참고용이며, 실제 투자 결정은 개인의 판단과 책임하에 이루어져야 합니다. 
                            과거 성과가 미래 수익을 보장하지 않으며, 모든 투자에는 원금 손실 위험이 있습니다.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    # API 키 없을 때 안내 메시지를 예쁘게 표시
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #fff3cd 0%, #fce4a6 100%);
                        border: 1px solid #ffc107;
                        border-radius: 15px;
                        padding: 30px;
                        text-align: center;
                        margin: 20px 0;
                        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
                    ">
                        <div style="font-size: 3rem; margin-bottom: 15px;">🔑</div>
                        <h3 style="color: #856404; margin: 0 0 15px 0;">
                            AI 분석 기능을 사용하려면 API 키가 필요합니다
                        </h3>
                        <p style="color: #664d03; margin: 0 0 20px 0; line-height: 1.6;">
                            OPENAI_API_KEY와 FMP_API_KEY를 환경변수로 설정해주세요.
                        </p>
                        <div style="
                            background: rgba(255,255,255,0.8);
                            border-radius: 10px;
                            padding: 15px;
                            text-align: left;
                            font-family: monospace;
                            color: #495057;
                            font-size: 0.9rem;
                        ">
                            <strong>설정 방법:</strong><br>
                            1. .env 파일을 프로젝트 루트에 생성<br>
                            2. 다음 내용 추가:<br>
                            &nbsp;&nbsp;OPENAI_API_KEY=your_openai_key<br>
                            &nbsp;&nbsp;FMP_API_KEY=your_fmp_key
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 전략별 상세 정보
            st.markdown("---")
            st.subheader("📋 전략 상세 정보")
            
            strategy_info = {
                'sma_crossover': {
                    'name': 'SMA 교차 전략',
                    'description': '단기 이동평균선이 장기 이동평균선을 상향돌파하면 매수, 하향돌파하면 매도하는 전략',
                    'pros': '• 트렌드 추종에 효과적\n• 이해하기 쉬운 로직\n• 강한 추세에서 큰 수익 가능',
                    'cons': '• 횡보장에서 잦은 거짓 신호\n• 지연된 신호로 인한 늦은 진입/청산'
                },
                'macd': {
                    'name': 'MACD 전략',
                    'description': 'MACD 라인이 시그널 라인을 상향돌파하면 매수, 하향돌파하면 매도하는 전략',
                    'pros': '• 빠른 신호 감지\n• 모멘텀 변화 포착에 효과적\n• 다양한 시장 상황에 적용 가능',
                    'cons': '• 변동성이 큰 시장에서 거짓 신호\n• 매개변수 설정에 민감'
                },
                'rsi': {
                    'name': 'RSI 전략',
                    'description': 'RSI가 과매도 구간에서 반등하면 매수, 과매수 구간에서 하락하면 매도하는 전략',
                    'pros': '• 과매수/과매도 구간 식별에 효과적\n• 단기 반전 포착\n• 리스크 관리에 유용',
                    'cons': '• 강한 트렌드에서 조기 진입/청산\n• 횡보장에서만 효과적'
                },
                'bollinger': {
                    'name': '볼린저 밴드 전략',
                    'description': '가격이 하단 밴드에서 반등하면 매수, 상단 밴드에서 하락하면 매도하는 전략',
                    'pros': '• 변동성을 고려한 매매\n• 평균회귀 전략에 효과적\n• 동적 지지/저항선 제공',
                    'cons': '• 강한 트렌드에서 밴드 이탈 지속\n• 변동성 확대 시 신호 지연'
                },
                'obv': {
                    'name': 'OBV 전략',
                    'description': 'OBV가 이동평균선을 상향돌파하면 매수, 하향돌파하면 매도하는 전략',
                    'pros': '• 거래량을 고려한 분석\n• 가격 움직임 선행 신호\n• 추세 전환점 포착',
                    'cons': '• 거래량 데이터의 정확성에 의존\n• 단독 사용 시 신뢰성 한계'
                },
                'combined': {
                    'name': '통합 전략',
                    'description': '5개 전략 중 3개 이상에서 동일한 신호가 나올 때만 매매하는 보수적 전략',
                    'pros': '• 거짓 신호 최소화\n• 여러 지표의 확인을 통한 신뢰성 향상\n• 리스크 분산',
                    'cons': '• 신호 발생 빈도 감소\n• 기회 손실 가능성\n• 복잡한 로직'
                }
            }
            
            info = strategy_info[selected_strategy]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**전략명:** {info['name']}")
                st.markdown(f"**설명:** {info['description']}")
                
                st.markdown("**장점:**")
                st.text(info['pros'])
            
            with col2:
                st.markdown("**단점:**")
                st.text(info['cons'])
                
                # 현재 전략 파라미터 표시
                st.markdown("**현재 파라미터:**")
                current_params = StockAnalyzer.STRATEGY_PARAMS.get(selected_strategy, {})
                for key, value in current_params.items():
                    st.text(f"• {key}: {value}")
                    
        except Exception as e:
            st.error(f"❌ 오류 발생: {str(e)}")
            st.markdown("**가능한 해결방법:**")
            st.markdown("• 유효한 주식 심볼을 입력했는지 확인")
            st.markdown("• 날짜 범위가 적절한지 확인")
            st.markdown("• 인터넷 연결 상태 확인")

else:
    # 초기 페이지 안내
    st.markdown("""
    ## 🎯 주식 자동매매 전략 분석 시스템 사용법
    
    ### 📋 설정 단계
    1. **왼쪽 사이드바**에서 분석할 종목의 심볼을 입력하세요 (예: AAPL, GOOGL)
    2. **분석 기간**을 설정하세요 (기본: 최근 3년)
    3. **초기 투자금**을 설정하세요 (기본: 1억원)
    4. **거래 전략**을 선택하세요
    5. 필요시 **파라미터 최적화** 옵션을 체크하세요
    
    ### 🚀 분석 실행
    - **"분석 시작"** 버튼을 클릭하여 백테스팅을 실행하세요
    - 결과는 성과 지표, 차트, AI 분석으로 구성됩니다
    
    ### 📊 제공되는 전략
    - **SMA 교차**: 이동평균선 교차 신호
    - **MACD**: 모멘텀 분석 기반 신호  
    - **RSI**: 과매수/과매도 구간 분석
    - **볼린저 밴드**: 변동성 기반 매매
    - **OBV**: 거래량 분석 기반 신호
    - **통합 전략**: 여러 전략의 합의 기반
    
    ### 🤖 AI 분석 기능
    - OpenAI GPT를 활용한 전문가 수준의 투자 분석
    - 시장 환경, 기업 펀더멘털, 전략 성과를 종합 분석
    - 구체적인 투자 의견과 리스크 분석 제공
    
    **시작하려면 왼쪽 설정을 완료하고 "분석 시작" 버튼을 눌러주세요! 🚀**
    """)
    
    # 샘플 이미지나 차트를 보여줄 수도 있습니다
    st.markdown("---")
    st.markdown("### 💡 주요 기능 미리보기")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📈 실시간 차트 분석**
        - 인터랙티브 차트
        - 매매 신호 표시
        - 기술적 지표 시각화
        """)
    
    with col2:
        st.markdown("""
        **📊 성과 지표**
        - 총수익률 vs 매수보유
        - 연간 수익률
        - 최대 낙폭 (MDD)
        - 거래 횟수
        """)
    
    with col3:
        st.markdown("""
        **🤖 AI 전문가 분석**
        - 투자 의견 및 근거
        - 시장 환경 분석
        - 리스크 요인 분석
        - 전략 추천
        """)
