import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- 網頁標題與設定 ---
st.set_page_config(page_title="IoT Botnet 偵測監控系統", layout="wide")
st.title("🛡️ 物聯網設備 (IoT) 惡意流量即時偵測系統")
st.markdown("本系統基於 **隨機森林 (Random Forest)** 演算法，針對 Bot-IoT 資料集進行即時分類與分析。")

# --- 1. 資料載入與模型預訓練 ---
@st.cache_resource # 讓模型只訓練一次，避免網頁重整時卡頓
def train_model():
    # 加上 nrows=50000 限制讀取行數，這能大幅節省記憶體並加快啟動速度
    # 必須先讀入 df，再調用 .sample() 方法
    df_full = pd.read_csv('archive/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv')
    df = df_full.sample(n=50000)
    # 資料處理
    if 'proto' in df.columns: df['proto'] = pd.factorize(df['proto'])[0]
    if 'state' in df.columns: df['state'] = pd.factorize(df['state'])[0]
    
    drop_cols = ['pkSeqID', 'saddr', 'daddr', 'sport', 'dport', 'attack', 'category', 'subcategory']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['attack']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf, X_test, y_test, df

model, X_test, y_test, raw_df = train_model()

# --- 2. 側邊欄：互動控制 ---
st.sidebar.header("控制面板")
sim_speed = st.sidebar.slider("模擬速度 (秒)", 0.5, 3.0, 1.0)
num_samples = st.sidebar.number_input("模擬樣本數", 5, 50, 10)

# --- 3. 視覺化圖表區 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 特徵重要性分析")
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    fig_imp, ax_imp = plt.subplots()
    importances.nlargest(10).plot(kind='barh', ax=ax_imp, color='skyblue')
    st.pyplot(fig_imp)

with col2:
    st.subheader("📈 流量狀態比例")
    fig_pie, ax_pie = plt.subplots()
    raw_df['attack'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Normal', 'Attack'], ax=ax_pie, colors=['green', 'red'])
    st.pyplot(fig_pie)

# --- 4. 即時模擬監控區 ---
st.divider()
st.subheader("🚀 即時流量監控模擬")

if st.button("開始即時監控演示"):
    # 隨機抽取正常與攻擊資料混合
    mock_data = pd.concat([
        raw_df[raw_df['attack']==0].sample(min(len(raw_df[raw_df['attack']==0]), num_samples//2)),
        raw_df[raw_df['attack']==1].sample(num_samples//2)
    ]).sample(frac=1)

    placeholder = st.empty() # 建立一個空位來動態更新表格
    
    results = []
    for i in range(len(mock_data)):
        row = mock_data.iloc[i]
        features = row[X_test.columns].values.reshape(1, -1)
        pred = model.predict(features)[0]
        
        # 紀錄結果
        status = "🔴 ATTACK" if pred == 1 else "🟢 NORMAL"
        res = {
            "時間": time.strftime("%H:%M:%S"),
            "序列號": int(row['seq']),
            "發送速率 (srate)": f"{row['srate']:.2f}",
            "標準差 (stddev)": f"{row['stddev']:.4f}",
            "判定結果": status
        }
        results.insert(0, res) # 新的放上面
        
        # 動態更新畫面
        with placeholder.container():
            if pred == 1:
                st.error(f"偵測到攻擊！ 序列號: {int(row['seq'])}")
            else:
                st.success(f"設備安全 序列號: {int(row['seq'])}")
            st.table(pd.DataFrame(results))
            
        time.sleep(sim_speed)