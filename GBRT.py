import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split  # 🔹 Thêm dòng này

# --- Bước 1: Đọc và tiền xử lý dữ liệu ---
file_path = "AQI-Air-Quality-HaNoi.csv"
df = pd.read_csv(file_path)

# Chuyển đổi cột ngày thành kiểu datetime
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df = df.sort_values(by="Date")

# Chuyển đổi dữ liệu AQI thành số, xử lý giá trị thiếu
aqi_columns = ["PM2.5_AQI", "PM10_AQI", "NO2_AQI", "SO2_AQI", "CO_AQI", "O3_AQI"]
df[aqi_columns] = df[aqi_columns].apply(pd.to_numeric, errors='coerce')
df[aqi_columns] = df[aqi_columns].interpolate()

# --- Bước 2: Tạo đặc trưng với 5 ngày trước ---
n_lags = 5
for i in range(1, n_lags + 1):
    for col in aqi_columns:
        df[f"{col}_lag_{i}"] = df[col].shift(i)
df = df.dropna()

# --- Bước 3: Tạo tập dữ liệu ---
features = [col for col in df.columns if "_lag_" in col]
targets = aqi_columns
X = df[features]
y = df[targets]

# 🔹 Chia dữ liệu thành tập huấn luyện & kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Bước 4: Huấn luyện mô hình GBRT ---
gbrt = MultiOutputRegressor(
    GradientBoostingRegressor(
        n_estimators=200,      # Số lượng cây quyết định
        learning_rate=0.05,    # Tốc độ học
        max_depth=5,           # Độ sâu tối đa của mỗi cây
        subsample=0.8,         # Giúp giảm overfitting
        random_state=42
    )
)
gbrt.fit(X_train, y_train)  # 🔹 Dùng tập huấn luyện thay vì toàn bộ dữ liệu

# --- Bước 5: Dự đoán và đánh giá ---
y_pred = gbrt.predict(X_test)  # 🔹 Dự đoán trên tập kiểm tra
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# --- Kết quả ---
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
