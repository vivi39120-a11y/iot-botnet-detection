import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. 讀取資料
print("正在讀取原始數據...")
df = pd.read_csv('archive/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv')

# --- 2. 資料預處理 (確保與部署端一致) ---
# 定義明確要丟棄的非特徵欄位
drop_cols = ['pkSeqID', 'saddr', 'daddr', 'sport', 'dport', 'attack', 'category', 'subcategory']
# 只保留存在的欄位
existing_drop_cols = [c for c in drop_cols if c in df.columns]

# 處理文字欄位 (使用 factorize 替代 LabelEncoder 以簡化部署)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.factorize(df[col].astype(str))[0]

# 設定特徵 (X) 與標籤 (y)
# 這裡以 'category' 為訓練目標
X = df.drop(columns=existing_drop_cols)
y = df['category']

# 紀錄精確的特徵順序清單
feature_list = X.columns.tolist()

# 3. 訓練模型
print(f"正在訓練幾百萬筆數據 (特徵數: {len(feature_list)})...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

# --- 4. 打包匯出 (核心修正：將模型與特徵清單封裝) ---
model_package = {
    'model': model,
    'features': feature_list
}
joblib.dump(model_package, 'iot_model.pkl')

print("✅ 模型與特徵清單匯出成功！檔案：iot_model.pkl")
print(f"訓練特徵順序: {feature_list}")