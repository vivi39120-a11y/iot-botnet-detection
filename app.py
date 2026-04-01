import streamlit as st
import joblib
import pandas as pd
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="IoT 安全專案", page_icon="🛡️", layout="wide")

# -----------------------------
# 1. 載入模型包
# -----------------------------
@st.cache_resource
def load_trained_assets():
    package = joblib.load("iot_model.pkl")
    return (
        package["model"],
        package["features"],
        package["categorical_cols"],
        package["numeric_cols"],
        package["encoder"],
        package["label_encoder"],
        package["drop_cols"]
    )

try:
    (
        model,
        trained_features,
        categorical_cols,
        numeric_cols,
        encoder,
        label_encoder,
        drop_cols
    ) = load_trained_assets()
except Exception as e:
    st.error(f"讀取模型失敗：{e}")
    st.stop()

# -----------------------------
# 2. 載入並清理資料
# -----------------------------
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv(
        "archive/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv",
        nrows=5000
    )

    display_df = df.copy()

    # 只保留訓練時使用的特徵欄位
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()

    # 若有缺欄，補上
    X = X.reindex(columns=trained_features, fill_value=0)

    # 類別欄位：套用訓練時的 encoder
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)

    if categorical_cols:
        X[categorical_cols] = encoder.transform(X[categorical_cols])

    # 數值欄位：強制轉數值
    for col in numeric_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # 全部補 0 並轉 float
    X = X.fillna(0).astype(float)

    return display_df, X

try:
    display_df, processed_df = load_and_clean_data()
except Exception as e:
    st.error(f"資料前處理失敗：{e}")
    st.stop()

# -----------------------------
# 3. UI
# -----------------------------
st.title("物聯網惡意流量偵測系統")
st.sidebar.header("控制面板")

sim_speed = st.sidebar.slider("模擬速度（秒）", 0.1, 2.0, 0.5)
num_samples = st.sidebar.number_input("樣本數", min_value=5, max_value=100, value=15, step=1)

col1, col2 = st.columns(2)

with col1:
    st.subheader("流量統計")

    fig, ax = plt.subplots()

    # 優先用 category，其次 attack
    if "category" in display_df.columns:
        display_df["category"].astype(str).value_counts().plot.pie(
            autopct='%1.1f%%',
            ax=ax
        )
    elif "attack" in display_df.columns:
        display_df["attack"].astype(str).value_counts().plot.pie(
            autopct='%1.1f%%',
            ax=ax
        )
    else:
        ax.text(0.5, 0.5, "找不到 category/attack 欄位", ha="center", va="center")
        ax.axis("off")

    st.pyplot(fig)

with col2:
    st.subheader("模型關鍵特徵")
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=trained_features)
        fig_bar, ax_bar = plt.subplots()
        importances.nlargest(5).sort_values().plot(kind="barh", ax=ax_bar)
        st.pyplot(fig_bar)
    else:
        st.info("此模型不支援特徵重要度顯示")

# -----------------------------
# 4. 監控演示
# -----------------------------
def get_seq_value(row):
    for key in ["pkSeqID", "seq", "Seq", "id"]:
        if key in row.index:
            return row[key]
    return "N/A"

if st.button("開始監控演示"):
    samples = processed_df.sample(n=int(num_samples), random_state=42)
    placeholder = st.empty()
    results_log = []

    for idx, row in samples.iterrows():
        input_data = row.values.reshape(1, -1)

        pred_encoded = model.predict(input_data)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        orig_row = display_df.loc[idx]
        seq_value = get_seq_value(orig_row)

        status = str(pred_label)

        # 你可以依資料集實際 normal 名稱再調整
        normal_keywords = ["normal", "benign"]
        is_attack = status.strip().lower() not in normal_keywords

        res = {
            "時間": time.strftime("%H:%M:%S"),
            "序列號": seq_value,
            "偵測類別": status,
            "判定": "🔴 ATTACK" if is_attack else "🟢 NORMAL"
        }
        results_log.insert(0, res)

        with placeholder.container():
            if is_attack:
                st.error(f"偵測到攻擊：{status}")
            else:
                st.success(f"流量正常：{status}")

            st.table(pd.DataFrame(results_log))

        time.sleep(sim_speed)