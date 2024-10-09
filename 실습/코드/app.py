import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time

# 페이지 설정
st.set_page_config(page_title="스마트 그린월 모니터링", layout="wide")

# 사이드바를 사용하여 페이지 선택
page = st.sidebar.selectbox("페이지 선택", ["실시간 모니터링", "제어 패널"])

# 데이터 불러오기
data = pd.read_csv(r'C:\Users\DC\OneDrive - 계명대학교\DC\2024\2024_개인\-\실습\데이터셋\open (1)\train_input\merged_data.csv')

# 공기 정화량 생성 함수
def generate_purification_data(size):
    return np.random.uniform(0, 50, size)

# 데이터프레임에 공기 정화량 추가
data['공기정화량'] = generate_purification_data(len(data))

# 데이터 저장을 위한 리스트 초기화
temperature_data = []
humidity_data = []
co2_data = []
purification_data = []

# 초기화된 상태가 없으면 session_state에 데이터 초기화
if "index" not in st.session_state:
    st.session_state.index = 0

if page == "실시간 모니터링":
    st.title("🌱 실시간 스마트 그린월 모니터링 대시보드")
    st.markdown("#### 온도, 습도, CO2 및 공기 정화량 모니터링")

    # 실시간 데이터 시뮬레이션
    placeholder = st.empty()

    # 데이터가 있는 동안 업데이트 반복
    while st.session_state.index < len(data):
        current_data = data.iloc[st.session_state.index]
        temperature_data.append(current_data['내부온도관측치'])
        humidity_data.append(current_data['내부습도관측치'])
        co2_data.append(current_data['co2관측치'])
        purification_data.append(current_data['공기정화량'])

        with placeholder.container():
            # 데이터 요약 표시
            st.markdown(f"""
            <div style="display: flex; justify-content: space-around; padding: 10px; background-color: #f0f0f0; border-radius: 10px;">
                <div><strong>🌡️ 온도:</strong> {temperature_data[-1]:.1f} ℃</div>
                <div><strong>💧 습도:</strong> {humidity_data[-1]:.1f} %</div>
                <div><strong>🌀 CO2 농도:</strong> {co2_data[-1]:.1f} ppm</div>
                <div><strong>🍃 공기 정화량:</strong> {purification_data[-1]:.1f} g</div>
            </div>
            """, unsafe_allow_html=True)

            # 그래프를 2개의 열로 배치하여 한눈에 보이도록 설정
            col1, col2 = st.columns(2)

            with col1:
                # 온도 그래프
                temp_fig = go.Figure(data=go.Scatter(
                    x=list(range(len(temperature_data))),
                    y=temperature_data,
                    mode='lines',
                    line=dict(color='red')
                ))
                temp_fig.update_layout(title="🌡️ 온도 (℃)", xaxis_title=None, showlegend=False)
                temp_fig.update_xaxes(visible=False)
                st.plotly_chart(temp_fig, use_container_width=True)

                # 습도 그래프
                humidity_fig = go.Figure(data=go.Scatter(
                    x=list(range(len(humidity_data))),
                    y=humidity_data,
                    mode='lines',
                    line=dict(color='skyblue')
                ))
                humidity_fig.update_layout(title="💧 습도 (%)", xaxis_title=None, showlegend=False)
                humidity_fig.update_xaxes(visible=False)
                st.plotly_chart(humidity_fig, use_container_width=True)

            with col2:
                # CO2 농도 그래프
                co2_fig = go.Figure(data=go.Scatter(
                    x=list(range(len(co2_data))),
                    y=co2_data,
                    mode='lines',
                    line=dict(color='navy')
                ))
                co2_fig.update_layout(title="🌀 CO2 농도 (ppm)", xaxis_title=None, showlegend=False)
                co2_fig.update_xaxes(visible=False)
                st.plotly_chart(co2_fig, use_container_width=True)

                # 공기 정화량 그래프 (영역 그래프)
                purification_fig = go.Figure(data=go.Scatter(
                    x=list(range(len(purification_data))),
                    y=purification_data,
                    mode='lines',
                    line=dict(color='green'),
                    fill='tozeroy'  # 영역 채우기 설정
                ))
                purification_fig.update_layout(title="🍃 공기 정화량 (g)", xaxis_title=None, showlegend=False)
                purification_fig.update_xaxes(visible=False)
                st.plotly_chart(purification_fig, use_container_width=True)

        # 인덱스 증가
        st.session_state.index += 1

        # 1초 지연
        time.sleep(1)

elif page == "제어 패널":
    st.title("🔧 스마트 그린월 제어 패널")
    st.markdown("<h3 style='text-align: center;'>온도 및 분무 설정</h3>", unsafe_allow_html=True)

    # 온도 조절 스위치
    st.markdown("<h4 style='color: red;'>🌡️ 온도 설정</h4>", unsafe_allow_html=True)
    temp_control = st.slider("", min_value=15, max_value=30, value=22, format="%d ℃")
    st.markdown(f"<p style='text-align: center; font-size: 20px;'>현재 설정된 온도: <strong>{temp_control} ℃</strong></p>", unsafe_allow_html=True)

    # 분무 스위치
    st.markdown("<h4 style='color: skyblue;'>💧 분무 기능 설정</h4>", unsafe_allow_html=True)
    mist_control = st.checkbox("분무 기능 활성화")
    if mist_control:
        st.success("💧 분무 기능이 활성화되었습니다.", icon="💧")
    else:
        st.warning("💧 분무 기능이 비활성화되었습니다.", icon="💧")
