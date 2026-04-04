import streamlit as st
import joblib
import pandas as pd
import time
import matplotlib.pyplot as plt
import random
from datetime import datetime
from zoneinfo import ZoneInfo
import plotly.express as px
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

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
        package.get("normal_labels", ["Normal"]),
        package.get("model_name", "RandomForestClassifier"),
        package.get("data_path", "Unknown"),
        package.get("metrics", {})
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
        data_path,
        metrics
    ) = load_trained_assets()
except Exception as e:
    st.error(f"讀取模型失敗：{e}")
    st.stop()

# -----------------------------
# 2. 側邊欄
# -----------------------------
st.sidebar.header("控制面板")
sim_speed = st.sidebar.slider("模擬速度（秒）", 0.1, 2.0, 0.5)
num_samples = st.sidebar.number_input("樣本數", min_value=5, max_value=100, value=20, step=1)
attack_ratio = st.sidebar.slider("模擬攻擊比例", min_value=0.05, max_value=0.50, value=0.15, step=0.05)
burst_mode = st.sidebar.checkbox("啟用攻擊爆發模式", value=True)

sample_size = st.sidebar.number_input("展示資料筆數", min_value=1000, max_value=20000, value=5000, step=1000)
fixed_sample = st.sidebar.checkbox("固定展示抽樣", value=False)
sample_seed = st.sidebar.number_input("固定抽樣種子", min_value=0, max_value=999999, value=42, step=1)

# -----------------------------
# 3. 載入並清理資料
# -----------------------------
@st.cache_data
def load_and_clean_data(sample_size, fixed_sample, sample_seed):
    df = pd.read_csv("archive/UNSW_NB15_testing-set.csv")

    if sample_size < len(df):
        if fixed_sample:
            df = df.sample(n=int(sample_size), random_state=int(sample_seed))
        else:
            df = df.sample(n=int(sample_size))

    df = df.reset_index(drop=True)
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
    display_df, processed_df = load_and_clean_data(sample_size, fixed_sample, sample_seed)
except Exception as e:
    st.error(f"資料前處理失敗：{e}")
    st.stop()

# -----------------------------
# 4. 工具函式
# -----------------------------
def get_seq_value(row):
    for key in ["id", "pkSeqID", "seq", "Seq"]:
        if key in row.index:
            return row[key]
    return "N/A"

def is_normal_label(label):
    text = str(label).strip().lower()
    valid = [str(x).strip().lower() for x in normal_labels]
    return text in valid or text in ["normal", "benign", "0"]

def pick_existing_columns(df, candidates):
    return [c for c in candidates if c in df.columns]

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None

def get_label_series(df):
    if "attack_cat" in df.columns:
        return df["attack_cat"].astype(str)
    if "label" in df.columns:
        return df["label"].astype(str)
    return None

# -----------------------------
# 5. 規則初篩（UNSW-NB15 版）
# -----------------------------
def rule_based_screening(orig_row):
    triggered_rules = []
    rule_score = 0

    dur = safe_float(orig_row["dur"]) if "dur" in orig_row.index else None
    rate = safe_float(orig_row["rate"]) if "rate" in orig_row.index else None
    spkts = safe_float(orig_row["spkts"]) if "spkts" in orig_row.index else None
    dpkts = safe_float(orig_row["dpkts"]) if "dpkts" in orig_row.index else None
    sbytes = safe_float(orig_row["sbytes"]) if "sbytes" in orig_row.index else None
    dbytes = safe_float(orig_row["dbytes"]) if "dbytes" in orig_row.index else None
    sload = safe_float(orig_row["sload"]) if "sload" in orig_row.index else None
    dload = safe_float(orig_row["dload"]) if "dload" in orig_row.index else None
    ct_srv_src = safe_float(orig_row["ct_srv_src"]) if "ct_srv_src" in orig_row.index else None
    ct_dst_src_ltm = safe_float(orig_row["ct_dst_src_ltm"]) if "ct_dst_src_ltm" in orig_row.index else None

    if rate is not None and rate > 1000:
        triggered_rules.append("High flow rate")
        rule_score += 2

    if sload is not None and sload > 1000000:
        triggered_rules.append("High source load")
        rule_score += 2

    if dload is not None and dload > 1000000:
        triggered_rules.append("High destination load")
        rule_score += 2

    if spkts is not None and spkts > 50:
        triggered_rules.append("Large source packet count")
        rule_score += 1

    if dpkts is not None and dpkts > 50:
        triggered_rules.append("Large destination packet count")
        rule_score += 1

    if dur is not None and sbytes is not None:
        if dur < 0.1 and sbytes > 10000:
            triggered_rules.append("Burst source traffic")
            rule_score += 2

    if dur is not None and dbytes is not None:
        if dur < 0.1 and dbytes > 10000:
            triggered_rules.append("Burst destination traffic")
            rule_score += 2

    if ct_srv_src is not None and ct_srv_src > 20:
        triggered_rules.append("High repeated service access")
        rule_score += 1

    if ct_dst_src_ltm is not None and ct_dst_src_ltm > 20:
        triggered_rules.append("High repeated src-dst activity")
        rule_score += 1

    return rule_score, triggered_rules

def get_risk_level(rule_score, is_attack, triggered_rules):
    if is_attack and rule_score >= 2:
        return "HIGH"
    if is_attack:
        return "MEDIUM"
    if rule_score >= 2 or len(triggered_rules) > 0:
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
# 6. 用原始標籤建立展示池
# -----------------------------
@st.cache_data
def build_display_pools(_display_df):
    label_series = get_label_series(_display_df)
    if label_series is None:
        all_idx = _display_df.index.tolist()
        return all_idx, all_idx

    normal_indices = label_series[label_series.apply(is_normal_label)].index.tolist()
    attack_indices = label_series[~label_series.apply(is_normal_label)].index.tolist()
    return normal_indices, attack_indices

try:
    normal_pool, attack_pool = build_display_pools(display_df)
except Exception as e:
    st.error(f"建立樣本池失敗：{e}")
    st.stop()

if not normal_pool:
    normal_pool = display_df.index.tolist()
if not attack_pool:
    attack_pool = display_df.index.tolist()


# -----------------------------
# 7. 頁面佈局與資料視覺化
# -----------------------------
st.title("物聯網惡意流量偵測系統")
st.caption("參考市面上 IoT 安全產品流程的簡易監控模擬：規則初篩 + 模型判斷 + 風險分級")
st.caption(f"模型：{model_name} ｜ 訓練資料：{data_path}")

# --- 第一層：資料集總覽 (左圖右表) ---
st.header("1. 資料集流量分布 (Dataset Overview)")
dist_col1, dist_col2 = st.columns([1.2, 1])

with dist_col1:
    # 稍微調整畫布大小
    fig, ax = plt.subplots(figsize=(7, 5)) 
    label_series = get_label_series(display_df)
    
    if label_series is not None:
        counts = label_series.value_counts()
        # 繪製經典圓餅圖
        counts.plot.pie(
            autopct="%1.1f%%", 
            ax=ax, 
            startangle=140, 
            pctdistance=0.75,    # 百分比數字的位置 (0-1 之間)
            labeldistance=1.1,   # 類別文字標籤的位置 (大於 1 會往圓外移)
            textprops={'fontsize': 10}
        )
        ax.set_ylabel("")
        
        # --- 關鍵修改：刪除以下這兩行，圓餅圖就不會變甜甜圈 ---
        # centre_circle = plt.Circle((0,0), 0.70, fc='white')
        # fig.gca().add_artist(centre_circle)
        
        # 強制緊湊佈局，防止標籤文字超出圖片邊界
        plt.tight_layout()
    else:
        ax.text(0.5, 0.5, "找不到標籤欄位", ha="center")
    
    st.pyplot(fig)

with dist_col2:
    st.write("#### 流量統計詳細資料")
    if label_series is not None:
        # 建立文字版比例表格
        dist_df = pd.DataFrame({
            "類別": counts.index,
            "樣本數": counts.values,
            "百分比": [f"{(v/counts.sum())*100:.1f}%" for v in counts.values]
        })
        st.table(dist_df) # 使用靜態表格呈現，方便閱讀
    else:
        st.info("暫無資料")

st.divider() # 畫一條橫向分隔線

# --- 第二層：模型分析 (全寬度展示) ---
st.header("2. 模型關鍵特徵 (Feature Importance)")

if hasattr(model, "feature_importances_"):
    # 1. 先定義對照表 (放在最前面確保後面抓得到)
    feature_name_map = {
        "dttl": "目的TTL",
        "service": "服務類型",
        "ct_dst_sport_ltm": "目的埠長期連線次數",
        "sbytes": "來源位元組數",
        "sttl": "來源TTL",
        "ct_srv_dst": "服務對應目的地次數",
        "smean": "來源封包平均大小",
        "ct_srv_src": "服務對應來源次數",
        "ct_state_ttl": "狀態與TTL組合次數",
        "ct_dst_src_ltm": "目的地與來源長期連線次數",
        "dur": "連線持續時間",
        "spkts": "來源封包數",
        "dpkts": "目的封包數",
        "dbytes": "目的位元組數",
        "rate": "流量速率",
        "sload": "來源負載",
        "dload": "目的負載",
        "sinpkt": "來源封包到達間隔時間",
        "dinpkt": "目的封包到達間隔時間",
        "tcprtt": "TCP連線往返時間",
        "synack": "TCP建立連線時間 (SYN-ACK)",
        "ackdat": "TCP資料傳輸延遲 (ACK-DAT)",
        "dmean": "目的封包平均大小",
        "ct_src_ltm": "同一來源地址長期連線次數",
        "ct_dst_ltm": "同一目的地址長期連線次數",
        "ct_src_dport_ltm": "來源與目的埠長期連線次數",
        "is_sm_ips_ports": "源與目的IP/埠是否相同",
        "swin": "來源TCP窗口大小",
        "dwin": "目的TCP窗口大小",
        "stcpb": "來源TCP序列號",
        "dtcpb": "目的TCP序列號",
        "trans_depth": "HTTP請求/響應深度"
    }

    # 2. 取得重要度數據
    importances = pd.Series(model.feature_importances_, index=trained_features)
    top_n = 12
    top_importances = importances.nlargest(top_n)

    # 3. 建立 DataFrame，同時轉換中文
    chart_data = pd.DataFrame({
        "特徵名稱": [feature_name_map.get(col, col) for col in top_importances.index],
        "重要度分數": top_importances.values
    }).sort_values("重要度分數", ascending=True)

    # 4. 使用 Plotly 繪圖
    fig_bar = px.bar(
        chart_data, 
        x="重要度分數", 
        y="特徵名稱", 
        orientation='h',
        text_auto='.3f',
        color="重要度分數",           # 增加顏色深淺變化，看起來更專業
        color_continuous_scale="Blues"
    )
    
    # 5. 關鍵設定：調整左邊距 (margin-l)，確保長文字不被切掉
    fig_bar.update_layout(
        margin=dict(l=200, r=20, t=20, b=20), # 如果字還是被切，就把 180 再調大
        yaxis={'title': ''},
        xaxis={'title': '重要度影響力'},
        showlegend=False,
        coloraxis_showscale=False # 隱藏右邊的顏色條，保持畫面簡潔
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("※ 數值越高代表該特徵對 AI 判斷「攻擊/正常」的影響力越大。")
else:
    st.info("目前的模型不支援顯示特徵重要度。")

summary_placeholder = st.empty()
status_placeholder = st.empty()

left_col, right_col = st.columns([1.5, 1])

with left_col:
    event_placeholder = st.empty()

with right_col:
    alert_placeholder = st.empty()

trend_placeholder = st.empty()
rank_placeholder = st.empty()

# -----------------------------
# 8. 模擬
# -----------------------------
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
        ["proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate"]
    )[:6]

    normal_queue = normal_pool.copy()
    attack_queue = attack_pool.copy()
    random.shuffle(normal_queue)
    random.shuffle(attack_queue)
    normal_ptr = 0
    attack_ptr = 0

    if int(num_samples) >= 10:
        burst_start = random.randint(5, max(5, int(num_samples) - 4))
        burst_end = min(burst_start + 3, int(num_samples))
    else:
        burst_start, burst_end = -1, -1

    for i in range(int(num_samples)):
        if burst_mode and burst_start <= i < burst_end:
            use_attack = random.random() < 0.75
        else:
            use_attack = random.random() < attack_ratio

        if use_attack:
            if attack_ptr >= len(attack_queue):
                random.shuffle(attack_queue)
                attack_ptr = 0
            idx = attack_queue[attack_ptr]
            attack_ptr += 1
        else:
            if normal_ptr >= len(normal_queue):
                random.shuffle(normal_queue)
                normal_ptr = 0
            idx = normal_queue[normal_ptr]
            normal_ptr += 1

        row = processed_df.iloc[idx]
        orig_row = display_df.iloc[idx]

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

        current_time = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%H:%M:%S")
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
        # 改成保留全部模擬結果
        results_log.insert(0, event_item)

        if risk_level in ["HIGH", "MEDIUM"]:
            alert_type = f"{risk_level} alert - {status}"
            rule_text = ", ".join(triggered_rules[:2]) if triggered_rules else "None"

            alert_log.insert(0, {
                "時間": current_time,
                "序列號": seq_value,
                "告警類型": alert_type,
                "命中規則": rule_text
            })

        # 改成全部測試過程的累積結果
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
            st.markdown(f"### 事件（共 {len(results_log)} 筆）")
            st.dataframe(
                pd.DataFrame(results_log),
                use_container_width=True,
                hide_index=True,
                height=420
            )

        with alert_placeholder.container():
            st.markdown("### 中高風險警示（共 {len(alert_log)} 筆）")

            if alert_log:
                alert_df = pd.DataFrame(alert_log)
            else:
                alert_df = pd.DataFrame([
                    {"時間": "", "序列號": "", "告警類型": "目前尚未出現中高風險告警", "命中規則": ""}
                ])

            st.dataframe(
                alert_df,
                use_container_width=True,
                hide_index=True,
                height=420,
                column_config={
                    "時間": st.column_config.TextColumn("時間", width="small"),
                    "序列號": st.column_config.TextColumn("序列號", width="small"),
                    "告警類型": st.column_config.TextColumn("告警類型", width="medium"),
                    "命中規則": st.column_config.TextColumn("命中規則", width="large"),
                }
            )

        with trend_placeholder.container():
            st.markdown("### 風險趨勢（全部測試結果）")
            chart_df = pd.DataFrame(stats_history).set_index("step")
            st.line_chart(chart_df[["Normal", "Low", "Medium", "High"]])
            st.caption("橫軸：測試筆數｜ 縱軸：累積事件數量")

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