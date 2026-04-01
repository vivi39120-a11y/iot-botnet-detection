import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. 讀取資料
print("正在讀取原始數據...")
df = pd.read_csv('archive/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv')

# --- 2. 嚴格預處理 (所有資料對齊的核心) ---
# 定義非特徵欄位（標籤與無關資訊）
drop_cols = ['pkSeqID', 'saddr', 'daddr', 'sport', 'dport', 'attack', 'category', 'subcategory']

# 遍歷所有欄位進行文字轉數字 (Encoding)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.factorize(df[col].astype(str))[0]

# 確保 X 只包含特徵，y 是目標
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df['category']

# 紀錄這份名單，這就是「對齊」的標準
feature_list = X.columns.tolist()

# 3. 訓練模型
print(f"訓練開始，特徵數量：{len(feature_list)}")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

# --- 4. 打包匯出 ---
model_package = {
    'model': model,
    'features': feature_list
}
joblib.dump(model_package, 'iot_model.pkl')

print("✅ 本地端訓練完成！請將 iot_model.pkl 上傳至 GitHub。")