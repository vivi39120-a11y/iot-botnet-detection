import streamlit as st
import joblib
import pandas as pd
import time
import matplotlib.pyplot as plt

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
    df = pd.read_csv('archive/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv', nrows=5000)
    display_df = df.copy()
    
    # 處理文字編碼 (必須與訓練時邏輯一致)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col].astype(str))[0]
            
    # 根據訓練時的清單進行「強制對齊」
    # 如果 CSV 缺少訓練時的欄位，這裡會補 0；如果多了，則會被捨棄
    aligned_df = df.reindex(columns=trained_features, fill_value=0)
    
    return display_df, aligned_df

display_df, processed_df = load_and_align_data()

# --- 3. UI 介面 ---
st.title("物聯網 (IoT) 惡意流量即時偵測系統")
st.success(f"模型對齊成功！已鎖定 {len(trained_features)} 個特徵欄位進行監控。")

col1, col2 = st.columns(2)
with col1:
    st.subheader("樣本統計")
    fig_pie, ax_pie = plt.subplots()
    display_df['attack'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Normal', 'Attack'], ax=ax_pie, colors=['#2ecc71', '#e74c3c'])
    st.pyplot(fig_pie)

with col2:
    st.subheader("核心特徵權衡")
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=trained_features)
        fig_bar, ax_bar = plt.subplots()
        importances.nlargest(5).plot(kind='barh', ax=ax_bar, color='skyblue')
        st.pyplot(fig_bar)

# --- 4. 即時監控模擬 ---
st.divider()
if st.button("開始執行監控演示"):
    samples = processed_df.sample(20)
    placeholder = st.empty()
    results_log = []
    
    for idx, row in samples.iterrows():
        # 這裡 row 的順序已經被 reindex 保證與訓練時一模一樣
        input_data = row.values.reshape(1, -1)
        pred = model.predict(input_data)[0]
        
        # 取得原始顯示數據
        orig_row = display_df.loc[idx]
        is_attack = (pred == 1 or str(pred).lower() == 'attack') # 容錯判斷
        
        res = {
            "時間": time.strftime("%H:%M:%S"),
            "序列號": int(orig_row['seq']),
            "協定": orig_row['proto'],
            "判別結果": "🔴 ATTACK" if is_attack else "🟢 NORMAL"
        }
        results_log.insert(0, res)
        
        with placeholder.container():
            if is_attack:
                st.error(f"偵測到入侵行為！ 序列號: {res['序列號']}")
            else:
                st.success(f"流量正常 序列號: {res['序列號']}")
            st.table(pd.DataFrame(results_log))
        
        time.sleep(0.5)