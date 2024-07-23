import streamlit as st
import pandas as pd
import numpy as np
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch
import matplotlib.pyplot as plt

# 모델과 feature extractor 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ViTForImageClassification.from_pretrained("imjeffhi/pokemon_classifier").to(device)
feature_extractor = ViTFeatureExtractor.from_pretrained('imjeffhi/pokemon_classifier')

# CSV 파일 경로 설정
df_path = 'Pokemon.csv'

# CSV 파일 미리 로드
try:
    df = pd.read_csv(df_path)
except FileNotFoundError:
    st.error(f"CSV 파일을 찾을 수 없습니다: {df_path}")
    st.stop()

# 사용자로부터 이미지 업로드 받기
st.title('포켓몬 도감 ◕‿◕✿')
uploaded_image = st.file_uploader("발견한 포켓몬을 찍어주세요!", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # 이미지를 열기
    img = Image.open(uploaded_image)
    st.image(img, caption='업로드한 이미지', use_column_width=True)

    try:
        # 이미지에서 특징 추출 및 예측
        extracted = feature_extractor(images=img, return_tensors='pt').to(device)
        predicted_id = model(**extracted).logits.argmax(-1).item()
        predicted_pokemon = model.config.id2label[predicted_id]
        st.write(f"예측된 포켓몬: {predicted_pokemon}")

        # 예측된 포켓몬에 해당하는 데이터 필터링
        def show_rows_by_value(df, column, value):
            result = df[df[column] == value]
            return result

        result = show_rows_by_value(df, 'Name', predicted_pokemon)
        st.write("포켓몬 정보:")

        if result.empty:
            st.write(f"포켓몬 {predicted_pokemon}의 정보가 없습니다. 정보를 입력해주세요.")
            
            hp = st.number_input('HP', min_value=0, max_value=255, step=1)
            attack = st.number_input('Attack', min_value=0, max_value=255, step=1)
            defense = st.number_input('Defense', min_value=0, max_value=255, step=1)
            sp_atk = st.number_input('Sp. Atk', min_value=0, max_value=255, step=1)
            sp_def = st.number_input('Sp. Def', min_value=0, max_value=255, step=1)
            speed = st.number_input('Speed', min_value=0, max_value=255, step=1)
            
            if st.button('정보 저장'):
                new_data = {
                    'Name': predicted_pokemon,
                    'HP': hp,
                    'Attack': attack,
                    'Defense': defense,
                    'Sp. Atk': sp_atk,
                    'Sp. Def': sp_def,
                    'Speed': speed
                }
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                df.to_csv(df_path, index=False)  # CSV 파일에 저장
                st.write("미지의 포켓몬을 발견 했습니다!!")
        else:
            st.write(result)

            # 각 스탯의 이름
            stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

            # 레이더 차트 그리기
            values = result[stats].values.flatten().tolist()
            values += values[:1]  # 레이더 차트를 닫기 위해 첫 번째 값을 다시 추가

            # 각 항목의 각도를 계산
            angles = np.linspace(0, 2 * np.pi, len(stats), endpoint=False).tolist()
            angles += angles[:1]  # 레이더 차트를 닫기 위해 첫 번째 각도를 다시 추가

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax.fill(angles, values, color='skyblue', alpha=0.25)
            ax.plot(angles, values, color='skyblue', linewidth=2)

            # 각 항목의 레이블 설정
            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(stats)

            plt.title(f'{predicted_pokemon} Stats Radar Chart')
            st.pyplot(fig)
    
    except Exception as e:
        st.write("미지의 포켓몬을 발견 했습니다!!. 포켓몬의 이름과 정보를 입력해주세요.")
        
        name = st.text_input('포켓몬 이름')
        hp = st.number_input('HP', min_value=0, max_value=255, step=1)
        attack = st.number_input('Attack', min_value=0, max_value=255, step=1)
        defense = st.number_input('Defense', min_value=0, max_value=255, step=1)
        sp_atk = st.number_input('Sp. Atk', min_value=0, max_value=255, step=1)
        sp_def = st.number_input('Sp. Def', min_value=0, max_value=255, step=1)
        speed = st.number_input('Speed', min_value=0, max_value=255, step=1)
        
        if st.button('정보 저장'):
            new_data = {
                'Name': name,
                'HP': hp,
                'Attack': attack,
                'Defense': defense,
                'Sp. Atk': sp_atk,
                'Sp. Def': sp_def,
                'Speed': speed
            }
            existing_pokemon = show_rows_by_value(df, 'Name', name)
            if existing_pokemon.empty:
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                message = "새로운 포켓몬이 추가되었습니다.🎉🎉"
            else:
                df.update(pd.DataFrame([new_data]))
                message = "미지의 포켓몬 정보를 업데이트했습니다.🎉🎉"
            df.to_csv(df_path, index=False)  # CSV 파일에 저장
            st.write(message)





# streamlit run 'C:\Users\Owner\OneDrive - 계명대학교\DC\2024\2024_개인\-\app.py'
# 6세대까지 인식 가능 + 정보 가지고 있음
# 7세대는 이름 O, 정보 X
# 8세대부터는 이름 X, 정보 X
