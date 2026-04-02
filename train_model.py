import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. 資料路徑
train_path = "archive/UNSW_NB15_training-set.csv"
test_path = "archive/UNSW_NB15_testing-set.csv"

print("正在讀取訓練資料...")
train_df = pd.read_csv(train_path)

print("正在讀取測試資料...")
test_df = pd.read_csv(test_path)

# 2. 設定標籤欄位與不作為特徵的欄位
target_col = "attack_cat"

# id 是流水號，不建議當特徵
# label 是二分類欄位，既然你要用 attack_cat，就不要放進特徵
drop_cols = ["id", "label", "attack_cat"]

# 3. 建立訓練 / 測試的 X, y
X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns]).copy()
y_train = train_df[target_col].astype(str).copy()

X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns]).copy()
y_test = test_df[target_col].astype(str).copy()

# 4. 找出類別欄位與數值欄位
categorical_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

# 5. 類別欄位先轉字串
for col in categorical_cols:
    X_train[col] = X_train[col].astype(str)
    if col in X_test.columns:
        X_test[col] = X_test[col].astype(str)

# 6. 類別欄位編碼
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

if categorical_cols:
    X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

# 7. 數值欄位轉換
for col in numeric_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    if col in X_test.columns:
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

# 8. 補缺值並轉 float
X_train = X_train.fillna(0).astype(float)
X_test = X_test.fillna(0).astype(float)

# 9. 標籤編碼
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# 10. 紀錄特徵順序
feature_list = X_train.columns.tolist()

# 11. 訓練模型
print(f"訓練開始，特徵數量：{len(feature_list)}")
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train_encoded)

# 12. 測試集評估
y_pred_encoded = model.predict(X_test)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n===== 模型評估結果（Testing Set）=====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\n===== 各類別詳細報告 =====")
print(classification_report(y_test, y_pred, zero_division=0))

# 13. 找出 normal 類別
possible_normal_labels = ["normal", "benign"]
normal_labels = [
    cls for cls in label_encoder.classes_
    if str(cls).strip().lower() in possible_normal_labels
]

# 14. 打包輸出
model_package = {
    "model": model,
    "model_name": "RandomForestClassifier",
    "data_path": train_path,
    "test_path": test_path,
    "target_col": target_col,
    "features": feature_list,
    "categorical_cols": categorical_cols,
    "numeric_cols": numeric_cols,
    "encoder": encoder,
    "label_encoder": label_encoder,
    "drop_cols": drop_cols,
    "class_names": list(label_encoder.classes_),
    "normal_labels": normal_labels,
    "metrics": {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }
}

joblib.dump(model_package, "iot_model.pkl", compress=3)

print("\n 本地端訓練完成！已輸出 iot_model.pkl")
print(f"類別數量：{len(label_encoder.classes_)}")
print(f"類別名稱：{list(label_encoder.classes_)}")
print(f"Normal 類別：{normal_labels if normal_labels else '未自動識別'}")