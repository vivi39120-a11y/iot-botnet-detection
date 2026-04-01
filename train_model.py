import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# 1. 讀取資料
data_path = "archive/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv"
print("正在讀取原始數據...")
df = pd.read_csv(data_path)

# 2. 定義非特徵欄位
drop_cols = ['pkSeqID', 'saddr', 'daddr', 'sport', 'dport', 'attack', 'category', 'subcategory']

# 3. 建立 X, y
X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
y = df['category'].astype(str).copy()

# 4. 找出類別欄位與數值欄位
categorical_cols = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
numeric_cols = [c for c in X.columns if c not in categorical_cols]

# 5. 類別欄位編碼
for col in categorical_cols:
    X[col] = X[col].astype(str)

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

if categorical_cols:
    X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# 6. 數值欄位轉換
for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.fillna(0).astype(float)

# 7. 標籤編碼
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 8. 紀錄特徵欄位順序
feature_list = X.columns.tolist()

# 9. 訓練模型
print(f"訓練開始，特徵數量：{len(feature_list)}")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y_encoded)

# 10. 推測 normal 類別名稱
possible_normal_labels = ["normal", "benign"]
normal_labels = [
    cls for cls in label_encoder.classes_
    if str(cls).strip().lower() in possible_normal_labels
]

# 11. 打包輸出
model_package = {
    'model': model,
    'model_name': 'RandomForestClassifier',
    'data_path': data_path,
    'features': feature_list,
    'categorical_cols': categorical_cols,
    'numeric_cols': numeric_cols,
    'encoder': encoder,
    'label_encoder': label_encoder,
    'drop_cols': drop_cols,
    'class_names': list(label_encoder.classes_),
    'normal_labels': normal_labels
}

joblib.dump(model_package, 'iot_model.pkl')
print("本地端訓練完成！已輸出 iot_model.pkl")
print(f"類別數量：{len(label_encoder.classes_)}")
print(f"類別名稱：{list(label_encoder.classes_)}")
print(f"Normal 類別：{normal_labels if normal_labels else '未自動識別'}")