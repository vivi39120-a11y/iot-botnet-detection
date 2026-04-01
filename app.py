import streamlit as st
import joblib
import pandas as pd
import time
import matplotlib.pyplot as plt

# 設定網頁標題
st.set_page_config(page_title="IoT 安全專題監控", page_icon="🛡️")

# --- 1. 載入模型包裹 ---
@st.cache_resource
def load_trained_assets():
    # 載入包含模型與特徵清單的字典
    package = joblib.load('iot_model.pkl')
    return package['model'], package['features']

try:
    model, trained_features = load_trained_assets()
except Exception as e:
    st.error(f"模型載入失敗，請確認已執行過 train_model.py 並上傳 pkl。錯誤: {e}")
    st.stop()

# --- 2. 載入展示數據 ---
@st.cache_data
def load_and_align_data():
    # 讀取測試資料
    df = pd.read_csv('archive/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv', nrows=5000)
    display_df = df.copy()
    
    # 處理文字編碼 (必須與訓練時邏輯一致)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col].astype(str))[0]
            
    # 根據訓練時的清單進行「強制對齊」
    # 保證欄位順序與訓練時 100% 一致
    aligned_df = df.reindex(columns=trained_features, fill_value=0)
    
    return display_df, aligned_df

display_df, processed_df = load_and_align_data()

# --- 3. UI 介面與側邊欄 ---
st.title("物聯網 (IoT) 惡意流量即時偵測系統")
st.success(f"模型對齊成功！已鎖定 {len(trained_features)} 個特徵欄位。")

# 側邊欄控制
st.sidebar.header("控制面板")
sim_speed = st.sidebar.slider("模擬速度 (秒)", 0.1, 2.0, 0.5)
# 這裡確保 num_samples 是整數
num_samples = st.sidebar.number_input("模擬樣本數", min_value=5, max_value=100, value=15)

col1, col2 = st.columns(2)
with col1:
    st.subheader("樣本統計")
    fig_pie, ax_pie = plt.subplots()
    display_df['attack'].value_counts().plot.pie(
        autopct='%1.1f%%', 
        labels=['Normal', 'Attack'], 
        ax=ax_pie, 
        colors=['#2ecc71', '#e74c3c']
    )
    st.pyplot(fig_pie)

with col2:
    st.subheader("核心特徵重要性")
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=trained_features)
        fig_bar, ax_bar = plt.subplots()
        importances.nlargest(5).plot(kind='barh', ax=ax_bar, color='skyblue')
        st.pyplot(fig_bar)

# --- 4. 即時監控模擬 ---
st.divider()
if st.button("開始執行監控演示"):
    # 修正：確保傳入 sample 的是整數 int
    samples_df = processed_df.sample(n=int(num_samples))
    placeholder = st.empty()
    results_log = []
    
    for idx, row in samples_df.iterrows():
        # --- 強制轉型修正 ---
        # 確保丟入模型的是 float 數值，避免 String to Float 錯誤
        input_data = row.values.astype(float).reshape(1, -1)
        
        # 2. 執行預測
        try:
            pred = model.predict(input_data)[0]
        except Exception as e:
            st.error(f"預測過程發生錯誤：{e}")
            st.stop()
        
        # 3. 取得原始對應的顯示數據
        orig_row = display_df.loc[idx]
        
        # 判斷結果
        status_text = str(pred)
        # 只要預測結果不是 0 或是 'normal'，就視為攻擊
        is_attack = status_text.lower() not in ['0', 'normal']
        
        res = {
            "時間": time.strftime("%H:%M:%S"),
            "序列號": int(orig_row['seq']),
            "協定": orig_row['proto'],
            "偵測類別": status_text,
            "狀態": "🔴 ATTACK" if is_attack else "🟢 NORMAL"
        }
        results_log.insert(0, res)
        
        # 4. 動態更新介面
        with placeholder.container():
            if is_attack:
                st.error(f"警報：偵測到 {status_text} 行為！ (序列號: {res['序列號']})")
            else:
                st.success(f"監測中：設備運行正常。 (序列號: {res['序列號']})")
            
            # 顯示結果表格
            st.table(pd.DataFrame(results_log))
        
        time.sleep(sim_speed)

st.info("提示：本系統採用隨機森林演算法，針對 Bot-IoT 資料集進行行為特徵偵測。")