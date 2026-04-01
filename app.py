import streamlit as st
import joblib
import pandas as pd
import time
import matplotlib.pyplot as plt

# --- 1. 載入預訓練大腦 ---
@st.cache_resource
def load_my_model():
    # 直接載入你本地端訓練好的那個最強大腦
    return joblib.load('iot_model.pkl')

model = load_my_model()

# --- 2. 載入少量數據供網頁「展示」使用 ---
@st.cache_data
def load_display_data():
    # 這裡只讀取 5000 筆，純粹是為了讓網頁有東西可以跑模擬演示
    df = pd.read_csv('archive/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv', nrows=5000)
    
    # 這裡必須做跟 train_model.py 一模一樣的特徵處理！
    # 如果你在訓練時有轉換文字，這裡也要轉，否則預測會噴錯
    if 'proto' in df.columns: df['proto'] = pd.factorize(df['proto'])[0]
    if 'state' in df.columns: df['state'] = pd.factorize(df['state'])[0]
    
    # 定義特徵欄位 (必須跟訓練時一模一樣)
    drop_cols = ['pkSeqID', 'saddr', 'daddr', 'sport', 'dport', 'attack', 'category', 'subcategory']
    features_cols = [c for c in df.columns if c not in drop_cols]
    
    return df, features_cols

raw_df, feature_cols = load_display_data()

# --- 3. 網頁介面 ---
st.title("🛡️ 物聯網 (IoT) 惡意流量即時偵測系統")
st.markdown(f"系統狀態：**已載入預訓練模型** (基於幾百萬筆數據訓練)")

# 側邊欄控制
st.sidebar.header("控制面板")
sim_speed = st.sidebar.slider("模擬速度 (秒)", 0.1, 2.0, 0.5)
num_samples = st.sidebar.number_input("模擬樣本數", 5, 50, 10)

# 視覺化圖表
col1, col2 = st.columns(2)
with col1:
    st.subheader("流量狀態比例")
    fig_pie, ax_pie = plt.subplots()
    raw_df['attack'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Normal', 'Attack'], ax=ax_pie, colors=['#2ecc71', '#e74c3c'])
    st.pyplot(fig_pie)

with col2:
    st.subheader("即時預警紀錄")
    # 這裡留空給下面的動態更新

# --- 4. 即時模擬監控區 ---
st.divider()
if st.button("開始即時監控演示"):
    # 隨機抽取展示用的資料
    mock_data = raw_df.sample(n=num_samples)
    placeholder = st.empty() 
    results = []
    
    for i in range(len(mock_data)):
        row = mock_data.iloc[i]
        
        # 準備特徵進行預測
        features = row[feature_cols].values.reshape(1, -1)
        pred = model.predict(features)[0]
        
        status = "🔴 ATTACK" if pred == 1 else "🟢 NORMAL"
        res = {
            "時間": time.strftime("%H:%M:%S"),
            "序列號": int(row['seq']),
            "發送速率 (srate)": f"{row['srate']:.2f}",
            "判定結果": status
        }
        results.insert(0, res) 
        
        with placeholder.container():
            if pred == 1:
                st.error(f"⚠️ 偵測到威脅！ 序列號: {int(row['seq'])}")
            else:
                st.success(f"✅ 設備運行正常 序列號: {int(row['seq'])}")
            st.table(pd.DataFrame(results))
            
        time.sleep(sim_speed)