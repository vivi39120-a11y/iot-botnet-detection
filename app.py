import streamlit as st
import joblib
import pandas as pd
import time
import matplotlib.pyplot as plt
import random

st.set_page_config(page_title="IoT 安全專案", layout="wide")

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
        package["drop_cols"],
        package.get("normal_labels", ["normal", "benign"]),
        package.get("model_name", "RandomForestClassifier"),
        package.get("data_path", "Unknown")
    )

try:
    (
        model,
        trained_features,
        categorical_cols,
        numeric_cols,
        encoder,
        label_encoder,
        drop_cols,
        normal_labels,
        model_name,
        data_path
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

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()
    X = X.reindex(columns=trained_features, fill_value=0)

    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)

    if categorical_cols:
        X[categorical_cols] = encoder.transform(X[categorical_cols])

    for col in numeric_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(0).astype(float)

    return display_df, X

try:
    display_df, processed_df = load_and_clean_data()
except Exception as e:
    st.error(f"資料前處理失敗：{e}")
    st.stop()


# -----------------------------
# 3. 工具函式
# -----------------------------
def get_seq_value(row):
    for key in ["pkSeqID", "seq", "Seq", "id"]:
        if key in row.index:
            return row[key]
    return "N/A"


def is_normal_label(label):
    return str(label).strip().lower() in [str(x).strip().lower() for x in normal_labels]


def pick_existing_columns(df, candidates):
    return [c for c in candidates if c in df.columns]


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


# -----------------------------
# 4. 模擬商業產品常見流程：規則初篩
# -----------------------------
def rule_based_screening(orig_row):
    """
    回傳:
    - rule_score: 規則風險分數
    - triggered_rules: 命中的規則列表
    """

    triggered_rules = []
    rule_score = 0

    # 可用欄位才檢查，避免子集欄位不齊時報錯
    dur = safe_float(orig_row["dur"]) if "dur" in orig_row.index else None
    rate = safe_float(orig_row["rate"]) if "rate" in orig_row.index else None
    srate = safe_float(orig_row["srate"]) if "srate" in orig_row.index else None
    drate = safe_float(orig_row["drate"]) if "drate" in orig_row.index else None
    pkts = safe_float(orig_row["pkts"]) if "pkts" in orig_row.index else None
    bytes_ = safe_float(orig_row["bytes"]) if "bytes" in orig_row.index else None

    # 規則 1：封包速率過高
    if rate is not None and rate > 1000:
        triggered_rules.append("High packet rate")
        rule_score += 2

    # 規則 2：來源速率過高
    if srate is not None and srate > 1000:
        triggered_rules.append("High source rate")
        rule_score += 2

    # 規則 3：目的速率過高
    if drate is not None and drate > 1000:
        triggered_rules.append("High destination rate")
        rule_score += 2

    # 規則 4：封包數異常大
    if pkts is not None and pkts > 50:
        triggered_rules.append("Large packet count")
        rule_score += 1

    # 規則 5：短時間大量流量
    if dur is not None and bytes_ is not None:
        if dur < 0.1 and bytes_ > 10000:
            triggered_rules.append("Burst traffic in short duration")
            rule_score += 2

    # 規則 6：非常短連線但高封包
    if dur is not None and pkts is not None:
        if dur < 0.05 and pkts > 20:
            triggered_rules.append("Short duration with many packets")
            rule_score += 2

    return rule_score, triggered_rules


def get_risk_level(rule_score, is_attack, triggered_rules):
    """
    風險分級：
    - High: 規則命中明顯 + 模型判攻擊
    - Medium: 模型判攻擊 或 規則偏高
    - Low: 有輕微異常
    - Normal: 無明顯異常
    """
    if is_attack and rule_score >= 2:
        return "HIGH"
    if is_attack:
        return "MEDIUM"
    if rule_score >= 2:
        return "LOW"
    if len(triggered_rules) > 0:
        return "LOW"
    return "NORMAL"


def get_risk_badge(level):
    if level == "HIGH":
        return "🔴 HIGH"
    if level == "MEDIUM":
        return "🟠 MEDIUM"
    if level == "LOW":
        return "🟡 LOW"
    return "🟢 NORMAL"


# -----------------------------
# 5. UI
# -----------------------------
st.title("物聯網惡意流量偵測系統")
st.caption("參考市面上 IoT 安全產品流程的簡易監控模擬：規則初篩 + 模型判斷 + 風險分級")
st.caption(f"模型：{model_name} ｜ 訓練資料：{data_path}")

st.sidebar.header("控制面板")
sim_speed = st.sidebar.slider("模擬速度（秒）", 0.1, 2.0, 0.5)
num_samples = st.sidebar.number_input("樣本數", min_value=5, max_value=100, value=20, step=1)
attack_ratio = st.sidebar.slider("模擬攻擊比例", min_value=0.05, max_value=0.50, value=0.15, step=0.05)
burst_mode = st.sidebar.checkbox("啟用攻擊爆發模式", value=True)

overview_col1, overview_col2 = st.columns(2)

with overview_col1:
    st.subheader("資料集流量分布")

    fig, ax = plt.subplots()

    if "category" in display_df.columns:
        display_df["category"].astype(str).value_counts().plot.pie(
            autopct="%1.1f%%",
            ax=ax
        )
        ax.set_ylabel("")
    elif "attack" in display_df.columns:
        display_df["attack"].astype(str).value_counts().plot.pie(
            autopct="%1.1f%%",
            ax=ax
        )
        ax.set_ylabel("")
    else:
        ax.text(0.5, 0.5, "找不到 category/attack 欄位", ha="center", va="center")
        ax.axis("off")

    st.pyplot(fig)

with overview_col2:
    st.subheader("模型關鍵特徵")
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=trained_features)
        fig_bar, ax_bar = plt.subplots()
        importances.nlargest(8).sort_values().plot(kind="barh", ax=ax_bar)
        ax_bar.set_xlabel("Importance")
        st.pyplot(fig_bar)
    else:
        st.info("此模型不支援特徵重要度顯示")


# -----------------------------
# 6. 預先建立正常 / 攻擊樣本池
# -----------------------------
@st.cache_data
def build_prediction_pools(_processed_df):
    all_pred_encoded = model.predict(_processed_df)

    if label_encoder is not None:
        all_pred_labels = label_encoder.inverse_transform(all_pred_encoded)
    else:
        all_pred_labels = all_pred_encoded

    pred_series = pd.Series(all_pred_labels, index=_processed_df.index)

    normal_indices = pred_series[pred_series.apply(is_normal_label)].index.tolist()
    attack_indices = pred_series[~pred_series.apply(is_normal_label)].index.tolist()

    return normal_indices, attack_indices

try:
    normal_pool, attack_pool = build_prediction_pools(processed_df)
except Exception as e:
    st.error(f"建立樣本池失敗：{e}")
    st.stop()

if not normal_pool:
    normal_pool = processed_df.index.tolist()
if not attack_pool:
    attack_pool = processed_df.index.tolist()


# -----------------------------
# 7. 即時監控區
# -----------------------------
st.subheader("即時監控模擬")

summary_placeholder = st.empty()
status_placeholder = st.empty()

left_col, right_col = st.columns([1.5, 1])

with left_col:
    event_placeholder = st.empty()

with right_col:
    alert_placeholder = st.empty()
    packet_placeholder = st.empty()

trend_placeholder = st.empty()
rank_placeholder = st.empty()

if st.button("開始監控演示"):
    total_count = 0
    normal_count = 0
    high_count = 0
    medium_count = 0
    low_count = 0

    results_log = []
    alert_log = []
    stats_history = []
    attack_type_counter = {}

    show_cols = pick_existing_columns(
        display_df,
        ["proto", "pkts", "bytes", "dur", "rate", "srate", "drate", "state"]
    )[:5]

    if int(num_samples) >= 10:
        burst_start = random.randint(5, max(5, int(num_samples) - 4))
        burst_end = min(burst_start + 3, int(num_samples))
    else:
        burst_start, burst_end = -1, -1

    for i in range(int(num_samples)):
        # 模擬一般情況 + 攻擊爆發時段
        if burst_mode and burst_start <= i < burst_end:
            use_attack = random.random() < 0.75
        else:
            use_attack = random.random() < attack_ratio

        idx = random.choice(attack_pool if use_attack else normal_pool)

        row = processed_df.loc[idx]
        orig_row = display_df.loc[idx]

        input_data = row.values.reshape(1, -1)
        pred_encoded = model.predict(input_data)[0]

        if label_encoder is not None:
            pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        else:
            pred_label = pred_encoded

        status = str(pred_label)
        is_attack = not is_normal_label(status)

        rule_score, triggered_rules = rule_based_screening(orig_row)
        risk_level = get_risk_level(rule_score, is_attack, triggered_rules)

        total_count += 1
        if risk_level == "NORMAL":
            normal_count += 1
        elif risk_level == "LOW":
            low_count += 1
        elif risk_level == "MEDIUM":
            medium_count += 1
        elif risk_level == "HIGH":
            high_count += 1

        current_time = time.strftime("%H:%M:%S")
        seq_value = get_seq_value(orig_row)

        if is_attack:
            attack_type_counter[status] = attack_type_counter.get(status, 0) + 1

        event_item = {
            "時間": current_time,
            "序列號": seq_value,
            "模型判斷": status,
            "規則分數": rule_score,
            "風險等級": get_risk_badge(risk_level),
            "最終判定": "ATTACK" if is_attack else "NORMAL"
        }
        results_log.insert(0, event_item)
        results_log = results_log[:15]

        if risk_level in ["HIGH", "MEDIUM"]:
            alert_text = f"{risk_level} alert - {status}"
            if triggered_rules:
                alert_text += f" | Rules: {', '.join(triggered_rules[:2])}"
            alert_log.insert(0, {
                "時間": current_time,
                "序列號": seq_value,
                "告警內容": alert_text
            })
            alert_log = alert_log[:8]

        stats_history.append({
            "step": total_count,
            "Normal": normal_count,
            "Low": low_count,
            "Medium": medium_count,
            "High": high_count
        })

        with summary_placeholder.container():
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("總流量", total_count)
            c2.metric("正常", normal_count)
            c3.metric("低風險", low_count)
            c4.metric("中風險", medium_count)
            c5.metric("高風險", high_count)

        with status_placeholder.container():
            if risk_level == "HIGH":
                st.error(f"【高風險告警】偵測到 {status}")
            elif risk_level == "MEDIUM":
                st.warning(f"【中風險告警】偵測到 {status}")
            elif risk_level == "LOW":
                st.info("偵測到輕微異常流量")
            else:
                st.success("目前流量正常")

        with event_placeholder.container():
            st.markdown("### 最近事件")
            st.table(pd.DataFrame(results_log))

        with alert_placeholder.container():
            st.markdown("### 最近告警")
            if alert_log:
                st.table(pd.DataFrame(alert_log))
            else:
                st.info("目前尚未出現中高風險告警")

        with packet_placeholder.container():
            st.markdown("### 最新流量摘要")
            packet_info = {
                "時間": current_time,
                "序列號": seq_value,
                "模型判斷": status,
                "規則分數": rule_score,
                "風險等級": risk_level,
                "命中規則": ", ".join(triggered_rules) if triggered_rules else "None"
            }

            for col in show_cols:
                packet_info[col] = orig_row[col]

            st.dataframe(pd.DataFrame([packet_info]), use_container_width=True, hide_index=True)

        with trend_placeholder.container():
            st.markdown("### 風險趨勢")
            chart_df = pd.DataFrame(stats_history).set_index("step")
            st.line_chart(chart_df[["Normal", "Low", "Medium", "High"]])

        with rank_placeholder.container():
            st.markdown("### 攻擊類型排行")
            if attack_type_counter:
                rank_df = (
                    pd.DataFrame(
                        [{"攻擊類型": k, "次數": v} for k, v in attack_type_counter.items()]
                    )
                    .sort_values("次數", ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(rank_df, use_container_width=True, hide_index=True)
            else:
                st.info("目前沒有攻擊類型資料")

        time.sleep(sim_speed)