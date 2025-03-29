import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
# --- Bước 1: Đọc và tiền xử lý dữ liệu ---
# Đọc file Excel. Bạn cần đảm bảo file "AirQualityUCI.xlsx" nằm cùng thư mục với file code.
df = pd.read_excel('AirQualityUCI.xlsx')

# Nếu có cột Date và Time, ta có thể kết hợp lại thành một cột datetime
if 'Date' in df.columns and 'Time' in df.columns:
    df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
    df = df.sort_values('Datetime')
    df = df.set_index('Datetime')
    
# Loại bỏ các cột không cần thiết nếu cần (ví dụ cột Date, Time ban đầu)
df = df.drop(columns=[col for col in ['Date', 'Time'] if col in df.columns])

# Xử lý giá trị thiếu (có thể dùng fillna hoặc dropna tùy dữ liệu)
df = df.dropna()

# --- Bước 2: Tạo dữ liệu chuỗi thời gian ---
# Giả sử dữ liệu đã được lấy theo ngày. Nếu dữ liệu theo giờ, bạn có thể cần resample (ví dụ: df = df.resample('D').mean())

# Các thông số: số ngày đầu vào và số ngày dự báo
input_window = 5       # Số ngày đầu vào (có thể thay đổi từ 5 đến 10)
forecast_horizon = 1   # Số ngày cần dự báo (có thể thay đổi từ 1 đến 3)

# Hàm tạo tập dữ liệu theo dạng chuỗi thời gian
def create_sequences(data, input_window, forecast_horizon):
    X, y = [], []
    # Giả sử dữ liệu sắp xếp theo thứ tự thời gian
    for i in range(len(data) - input_window - forecast_horizon + 1):
        # Lấy dữ liệu đầu vào là các ngày liên tiếp (input_window ngày)
        seq_x = data.iloc[i:i+input_window].values.flatten()  # Flatten để thành vector 1 chiều
        # Lấy dữ liệu dự báo là các ngày tiếp theo (forecast_horizon ngày)
        seq_y = data.iloc[i+input_window:i+input_window+forecast_horizon].values.flatten()
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Tạo các tập dữ liệu từ DataFrame
X, y = create_sequences(df, input_window, forecast_horizon)

print(f"Số mẫu tạo được: {X.shape[0]}")
print(f"Kích thước mỗi mẫu X: {X.shape[1]}, mỗi mẫu y: {y.shape[1]}")

# --- Bước 3: Huấn luyện và đánh giá mô hình ---
# Tách dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Khởi tạo mô hình GBRT trong MultiOutputRegressor
gbrt = MultiOutputRegressor(
    GradientBoostingRegressor(
        n_estimators=200,      # Số lượng cây quyết định (tăng độ chính xác)
        learning_rate=0.05,    # Tốc độ học nhỏ hơn để tránh overfitting
        max_depth=5,           # Độ sâu tối đa của mỗi cây
        subsample=0.8,         # Giúp giảm overfitting bằng cách lấy mẫu dữ liệu
        random_state=42
    )
)

gbrt.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = gbrt.predict(X_test)

# Tính độ chính xác theo hệ số R²
r2 = r2_score(y_test, y_pred)
accuracy_percentage = r2 * 100

# --- Bước 4: In ra dự đoán và độ chính xác ---
print("\nDự đoán cho tập kiểm tra:")
print(y_pred)

print(f"\nĐộ chính xác của mô hình (R²): {accuracy_percentage:.2f}%")
