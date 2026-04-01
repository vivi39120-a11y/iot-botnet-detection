import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. 讀取資料
df = pd.read_csv('archive/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv')

# --- 關鍵修正：處理文字欄位 ---
# 找出所有是「物件(文字)」類型的欄位
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col].astype(str))
# ----------------------------

# 2. 設定特徵 (X) 與標籤 (y)
# 請確認 'category' 是你的目標欄位，如果不對請修改名稱
X = df.drop(['category'], axis=1) 
y = df['category']

# 3. 訓練模型
print("正在訓練幾百萬筆數據，請稍候...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 會動用所有 CPU 核心加速
model.fit(X, y)

# 4. 匯出模型
joblib.dump(model, 'iot_model.pkl')
print("✅ 模型匯出成功！檔案：iot_model.pkl")