import streamlit as st
import joblib
import pandas as pd
import time
import matplotlib.pyplot as plt

# 設定網頁標題與圖示
st.set_page_config(page_title="IoT 惡意流量偵測系統", page_icon="🛡️")

# --- 1. 載入預訓練大腦 ---
@st.cache_resource
def load_my_model():
    # 載入你本地端訓練好的 .pkl 檔
    return joblib.load('iot_model.pkl')

try:
    model = load_my_model()
except FileNotFoundError:
    st.error("找不到模型檔 'iot_model.pkl'，請確保檔案已上傳至 GitHub 根目錄。")
    st.stop()

# --- 2. 載入與處理數據 ---
@st.cache_data
def load_display_data():
    # 讀取少量數據供網頁展示
    df = pd.read_csv('archive/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv', nrows=5000)
    
    # 備份原始資料供顯示用（避免顯示時全是數字）
    display_df = df.copy()
    
    # --- 關鍵修正：特徵處理必須與 train_model.py 完全一致 ---
    # 將所有文字欄位轉為數字
    for col in df.columns:
        if df[col].dtype == 'object':
            # 使用 factorize 確保文字轉成數字，這與 LabelEncoder 效果類似
            df[col] = pd.factorize(df[col].astype(str))[0]
    
    # 定義要丟棄的欄位 (這些通常是標籤或無關特徵)
    # 這裡請確保與你 train_model.py 中 drop 的內容完全相同
    drop_cols = ['pkSeqID', 'saddr', 'daddr', 'sport', 'dport', 'attack', 'category', 'subcategory']
    
    # 自動抓取剩餘的欄位作為特徵
    # 為了保險，我們直接根據模型的特徵數量來對齊
    available_features = [c for c in df.columns if c not in drop_cols]
    
    # 如果欄位多於模型要求的，只取前 N 個 (N 為模型訓練時的特徵數)
    # model.n_features_in_ 是 sklearn 內建屬性，紀錄訓練時的特徵數
    final_feature_cols = available_features[:model.n_features_in_]
    
    return display_df, df, final_feature_cols

try:
    display_df, processed_df, feature_cols = load_display_data()
except Exception as e:
    st.error(f"資料讀取失敗: {e}")
    st.stop()

# --- 3. 網頁介面設計 ---
st.title("物聯網 (IoT) 惡意流量即時偵測模擬系統")
st.markdown(f"系統狀態：**已載入預訓練模型** (模型特徵數: `{model.n_features_in_}`)")

# 側邊欄：控制面版
st.sidebar.header("控制面板")
sim_speed = st.sidebar.slider("模擬速度 (秒)", 0.1, 2.0, 0.5)
num_samples = st.sidebar.number_input("模擬樣本數", 5, 100, 15)

# 儀表板視覺化
col1, col2 = st.columns(2)
with col1:
    st.subheader("當前流量比例")
    fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
    display_df['attack'].value_counts().plot.pie(
        autopct='%1.1f%%', 
        labels=['Normal', 'Attack'], 
        ax=ax_pie, 
        colors=['#2ecc71', '#e74c3c'],
        startangle=90
    )
    st.pyplot(fig_pie)

with col2:
    st.subheader("模型特徵貢獻度")
    # 顯示前 5 名最重要的特徵
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        fig_bar, ax_bar = plt.subplots(figsize=(5, 5))
        importances.nlargest(5).sort_values().plot(kind='barh', ax=ax_bar, color='skyblue')
        st.pyplot(fig_bar)

# --- 4. 即時監控模擬區 ---
st.divider()
st.subheader("即時流量監控儀表")

if st.button("開始執行即時監控"):
    # 從處理過的數據中隨機抽樣
    sample_indices = processed_df.sample(n=num_samples).index
    placeholder = st.empty()
    log_results = []
    
    for idx in sample_indices:
        # 1. 取得該行數據
        row_processed = processed_df.loc[idx]
        row_display = display_df.loc[idx]
        
        # 2. 準備特徵進行預測 (確保形狀為 1 x N)
        features = row_processed[feature_cols].values.reshape(1, -1)
        
        # 3. 執行預測
        prediction = model.predict(features)[0]
        
        # 4. 判斷結果
        is_attack = (prediction == 1)
        status_icon = "🔴 ATTACK" if is_attack else "🟢 NORMAL"
        
        # 5. 紀錄到清單
        res = {
            "時間": time.strftime("%H:%M:%S"),
            "序列號": int(row_display['seq']),
            "協定": row_display['proto'],
            "速率 (srate)": f"{row_display['srate']:.2f}",
            "結果": status_icon
        }
        log_results.insert(0, res)
        
        # 6. 更新 UI
        with placeholder.container():
            if is_attack:
                st.error(f"警告：偵測到異常流量！ 序列號: {int(row_display['seq'])}")
            else:
                st.success(f"安全：設備運行正常。 序列號: {int(row_display['seq'])}")
            
            # 顯示紀錄表格
            st.table(pd.DataFrame(log_results))
            
        time.sleep(sim_speed)

st.info("提示：本演示從測試數據集中隨機抽樣，並利用預訓練模型進行推論。")