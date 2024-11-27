import streamlit as st
import pandas as pd
import requests
import datetime
import matplotlib.pyplot as plt
import folium
from shapely.geometry import Polygon
import geopandas as gpd
from streamlit_folium import st_folium
from matplotlib.colors import to_hex
import numpy as np

# 기본 설정
plt.rcParams['font.family'] = 'NanumGothic'
st.set_page_config(layout="wide", page_title="옹이구멍 관리 시스템")

# 카카오 API 키 설정 (자신의 카카오 API 키로 변경 필요)
KAKAO_API_KEY = "462b0af927ad676b7f8052a64d12ddff"

# 로그인 상태 관리
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['is_admin'] = False

def login():
    st.title("옹이구멍")
    st.markdown("### 포트홀 관리 시스템")
    
    user_id = st.text_input("아이디를 입력하세요")
    password = st.text_input("비밀번호를 입력하세요", type="password")
    
    if st.button("로그인"):
        if password == "0000":
            st.session_state['logged_in'] = True
            if user_id.startswith("1"):
                st.session_state['is_admin'] = True
            else:
                st.session_state['is_admin'] = False
        else:
            st.error("잘못된 비밀번호입니다.")

if not st.session_state['logged_in']:
    login()
else:
    # 그리드 파일 로드
    grid_df = pd.read_csv('C:/Users/Owner/OneDrive - 계명대학교/DC/2024/2024_개인/-/대구광역시그리드250.csv')
    dmddo2_df = pd.read_csv('C:/Users/Owner/OneDrive - 계명대학교/DC/2024/2024_개인/-/dmddo2.csv')
    krukum_df = pd.read_csv('C:/Users/Owner/OneDrive - 계명대학교/DC/2024/2024_개인/-/크큼.csv')

    # 임시 DB 초기화
    if 'temp_db' not in st.session_state:
        st.session_state['temp_db'] = pd.DataFrame(columns=['날짜', '주소', '그리드'])

    # 클릭 횟수 초기화
    if 'click_count' not in st.session_state:
        st.session_state['click_count'] = 0

    # 카카오 좌표 기반 주소 검색 API 호출 함수
    def get_address_by_coordinates(lon, lat):
        url = f"https://dapi.kakao.com/v2/local/geo/coord2address.json?x={lon}&y={lat}"
        headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            result = response.json()
            if len(result['documents']) > 0:
                address = result['documents'][0]['address']['address_name']
                return address
        return None

    # 위경도가 특정 그리드에 포함되는지 확인하는 함수
    def find_grid_id(lon, lat, grid_df):
        for _, row in grid_df.iterrows():
            if row['left'] <= lon <= row['right'] and row['top'] >= lat >= row['bottom']:
                return row['id']
        return None

    # 현재 위치를 가져오는 함수 (실제 구현 시 브라우저에서 위치 정보를 가져오는 방식 사용)
    def get_current_location():
        lon = 128.601445
        lat = 35.871435
        return lon, lat

    # 임의의 위치(구지면 특정 위치)를 가져오는 함수
    def get_random_location():
        lon = 128.422641  # 대구광역시 구지면의 임의의 경도
        lat = 35.692301  # 대구광역시 구지면의 임의의 위도
        return lon, lat

    # 관리자용 페이지들 (모든 페이지 접근 가능)
    if st.session_state['is_admin']:
        page = st.sidebar.selectbox('페이지 선택', ['포트홀 신고', '실무자 알림', '시각화', '예측'])
    else:
        page = '포트홀 신고'  # 일반 사용자는 포트홀 신고 페이지로만 접근 가능

    # 포트홀 신고 페이지
    if page == '포트홀 신고':
        st.title('포트홀 신고 시스템')
        st.markdown("도로에 포트홀이 발생하면 신속히 신고해 주세요. 신고된 정보는 관련 부서에서 처리됩니다.")

        st.divider()

        if st.button('현재 위치로 신고'):
            st.session_state['click_count'] += 1

            if st.session_state['click_count'] == 1:
                # 첫 번째 클릭: 현재 위치 사용
                lon, lat = get_current_location()
                st.info("현재 위치를 찾는 중입니다...")
            else:
                # 두 번째 클릭: 임의의 위치 사용
                lon, lat = get_random_location()
                st.info("현재 위치를 찾는 중입니다...")

            if lon and lat:
                # 위경도를 통해 주소 가져오기
                address = get_address_by_coordinates(lon, lat)

                if address:
                    # 위경도가 포함된 그리드 ID 찾기
                    grid_id = find_grid_id(lon, lat, grid_df)

                    if grid_id:
                        # 임시 DB에 저장
                        new_data = pd.DataFrame({'날짜': [datetime.datetime.now().date()], '주소': [address], '그리드': [grid_id]})
                        st.session_state.temp_db = pd.concat([st.session_state.temp_db, new_data], ignore_index=True)
                        st.success(f"포트홀 신고가 접수되었습니다. 주소: **{address}**, 그리드 ID: **{grid_id}**")
                    else:
                        st.warning("해당 위치는 그리드 내에 포함되지 않습니다.")
                else:
                    st.error("위치 정보를 가져오지 못했습니다. 다시 시도해 주세요.")
            else:
                st.error("위치 정보를 가져오지 못했습니다. 다시 시도해 주세요.")

        # 임시 DB 출력
        st.subheader('신고된 포트홀 목록')
        st.dataframe(st.session_state.temp_db)

    # 실무자 알림 페이지 (관리자 전용)
    elif page == '실무자 알림':
        st.title('실무자 알림 시스템')
        st.markdown("실무자들에게 현재 포트홀 발생 상황을 알리고 조치를 요청할 수 있습니다.")

        st.divider()

        # 임시 DB 출력
        st.subheader('신고된 포트홀 목록')
        st.dataframe(st.session_state.temp_db)

        # dmddo2.csv 파일의 그리드 ID와 비교
        for index, row in st.session_state.temp_db.iterrows():
            if row['그리드'] in dmddo2_df['grid_id'].values:
                st.warning(f"포트홀이 발생하지 않았던 그리드 ID {row['그리드']}에서 포트홀 발생, 집중 순찰 부탁드립니다.")

    # 시각화 페이지 (관리자 전용)
    elif page == '시각화':
        st.title('포트홀 시각화 시스템')
        st.markdown("선택된 기간 동안 발생한 포트홀을 지도에서 시각화합니다.")

        st.divider()

        # 연도와 월, 구 선택을 위한 selectbox 설정
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            year = st.selectbox('연도 선택', sorted(krukum_df['연도'].unique()))
        with col2:
            month = st.selectbox('월 선택', list(range(1, 13)))
        with col3:
            district = st.selectbox('구 선택', sorted(krukum_df['구'].unique()))

        # 데이터 필터링
        filtered_krukum = krukum_df[(krukum_df['연도'] == year) & (krukum_df['월'] == month) & (krukum_df['구'] == district)]

        # 좌표 참조 시스템(CRS) 설정 (예: UTM에서 EPSG:4326으로 변환)
        crs = "EPSG:4326"
        filtered_krukum = gpd.GeoDataFrame(filtered_krukum, geometry=gpd.points_from_xy(filtered_krukum['left'], filtered_krukum['top']), crs=crs)

        # 맵 생성
        m = folium.Map(location=[35.871435, 128.601445], zoom_start=11)

        # 그리드별 포트홀 발생 수에 따라 색상 설정 (원래 색깔로 설정)
        for _, row in filtered_krukum.iterrows():
            grid_polygon = Polygon([(row['left'], row['top']), 
                                    (row['right'], row['top']), 
                                    (row['right'], row['bottom']), 
                                    (row['left'], row['bottom'])])

            grid_polygon = gpd.GeoSeries([grid_polygon], crs=crs)  # CRS를 설정

            # 발생 수에 따라 원래 색상 지정
            color = plt.cm.Reds(row['포트홀 발생수'] / filtered_krukum['포트홀 발생수'].max())
            folium.GeoJson(
                grid_polygon,
                style_function=lambda x, color=to_hex(color): {'fillColor': color, 'color': color, 'weight': 1, 'fillOpacity': 0.6}
            ).add_to(m)

        # 맵 출력
        st_folium(m, width=900, height=500)

        # 지도 제목
        st.subheader(f"{year}년 {month}월 {district} 포트홀 발생 지도")

    # 예측 페이지 (관리자 전용)
    elif page == '예측':
        st.title('포트홀 예측 시스템')
        st.markdown("향후 포트홀 발생 가능성을 예측하고 시각화합니다.")

        st.divider()

        # 다음 1~3년 후와 월 선택을 위한 selectbox 설정
        col1, col2, col3 = st.columns([1, 1, 1])
        
        current_year=2024
        
        with col1:
            future_year = st.selectbox('예측 연도 선택', list(range(current_year + 1, current_year + 4)))
        with col2:
            month = st.selectbox('월 선택', list(range(1, 13)))
        with col3:
            district = st.selectbox('구 선택', sorted(krukum_df['구'].unique()))

        # 데이터 필터링 (현재 연도의 데이터를 기반으로 예측 확률 계산)
        filtered_krukum = krukum_df[(krukum_df['연도'] == current_year) & (krukum_df['월'] == month) & (krukum_df['구'] == district)]

        # 포트홀 예측 확률 계산 (연도별로 확률에 약간의 변동 추가)
        np.random.seed(42)  # 결과 재현성을 위한 시드 설정
        filtered_krukum['포트홀_예측_확률'] = filtered_krukum['포트홀 발생수'].apply(lambda x: 0.1 if x == 0 else min(0.1 + x * 0.1 * (future_year - current_year) * np.random.uniform(0.8, 1.2), 1.0))

        # 좌표 참조 시스템(CRS) 설정 (예: UTM에서 EPSG:4326으로 변환)
        crs = "EPSG:4326"
        filtered_krukum = gpd.GeoDataFrame(filtered_krukum, geometry=gpd.points_from_xy(filtered_krukum['left'], filtered_krukum['top']), crs=crs)

        # 포트홀 예측 확률 기반 지도 생성 (원래 색깔로 설정)
        m_prob = folium.Map(location=[35.871435, 128.601445], zoom_start=11)

        for _, row in filtered_krukum.iterrows():
            grid_polygon = Polygon([(row['left'], row['top']), 
                                    (row['right'], row['top']), 
                                    (row['right'], row['bottom']), 
                                    (row['left'], row['bottom'])])

            grid_polygon = gpd.GeoSeries([grid_polygon], crs=crs)  # CRS를 설정

            # 예측 확률에 따라 원래 색상 지정
            color = plt.cm.Blues(row['포트홀_예측_확률'])
            folium.GeoJson(
                grid_polygon,
                style_function=lambda x, color=to_hex(color): {'fillColor': color, 'color': color, 'weight': 1, 'fillOpacity': 0.6}
            ).add_to(m_prob)

        # 맵 출력 (예측 확률 지도)
        st_folium(m_prob, width=900, height=500)

        # 지도 제목
        st.subheader(f"{future_year}년 {month}월 {district} 포트홀 발생 확률 지도")
