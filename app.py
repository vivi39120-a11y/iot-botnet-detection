import streamlit as st
import joblib
import pandas as pd
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="IoT 安全專案", page_icon="🛡️")

# --- 1. 載入模型包裹 ---
@st.cache_resource
def load_trained_assets():
    package = joblib.load('iot_model.pkl')
    return package['model'], package['features']

try:
    model, trained_features = load_trained_assets()
except Exception as e:
    st.error(f"讀取模型失敗：{e}")
    st.stop()

# --- 2. 載入與強制對齊數據 ---
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('archive/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv', nrows=5000)
    display_df = df.copy()
    
    # 對所有欄位進行與訓練端一致的編碼
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col].astype(str))[0]
            
    # 【關鍵對齊】只取訓練時的那幾欄，順序必須完全一樣
    aligned_df = df.reindex(columns=trained_features, fill_value=0)
    
    # 【型態防護】強制轉為浮點數，確保 predict 不會噴錯
    aligned_df = aligned_df.astype(float)
    
    return display_df, aligned_df

display_df, processed_df = load_and_clean_data()

# --- 3. UI 介面 ---
st.title("物聯網惡意流量偵測系統")
st.sidebar.header("控制面板")
sim_speed = st.sidebar.slider("模擬速度", 0.1, 2.0, 0.5)
num_samples = st.sidebar.number_input("樣本數", 5, 100, 15)

col1, col2 = st.columns(2)
with col1:
    st.subheader("流量統計")
    fig, ax = plt.subplots()
    display_df['attack'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=['#2ecc71', '#e74c3c'])
    st.pyplot(fig)

with col2:
    st.subheader("模型關鍵特徵")
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=trained_features)
        fig_bar, ax_bar = plt.subplots()
        importances.nlargest(5).plot(kind='barh', ax=ax_bar)
        st.pyplot(fig_bar)

# --- 4. 監控演示 ---
if st.button("開始監控演示"):
    # 確保抽樣數是整數
    samples = processed_df.sample(n=int(num_samples))
    placeholder = st.empty()
    results_log = []
    
    for idx, row in samples.iterrows():
        # 因為前面已經 astype(float)，這裡直接 input
        input_data = row.values.reshape(1, -1)
        pred = model.predict(input_data)[0]
        
        orig_row = display_df.loc[idx]
        status = str(pred)
        is_attack = status.lower() not in ['0', 'normal']
        
        res = {
            "時間": time.strftime("%H:%M:%S"),
            "序列號": int(orig_row['seq']),
            "偵測類別": status,
            "判定": "🔴 ATTACK" if is_attack else "🟢 NORMAL"
        }
        results_log.insert(0, res)
        
        with placeholder.container():
            if is_attack:
                st.error(f"偵測到攻擊：{status}")
            else:
                st.success("流量正常")
            st.table(pd.DataFrame(results_log))
        time.sleep(sim_speed)