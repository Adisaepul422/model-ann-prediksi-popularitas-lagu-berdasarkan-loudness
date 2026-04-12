# train_loudness.py - Training Linear Regression dengan 1 fitur (loudness)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

print("="*60)
print("TRAINING MODEL PREDIKSI POPULARITAS (BERDASARKAN LOUDNESS)")
print("="*60)

# 1. Baca CSV
print("\n[1] Membaca file CSV...")
df = pd.read_csv('data_loudness.csv')
print(f"Data: {len(df)} lagu")
print(df.head(10))

# 2. Pisahkan X (loudness) dan y (popularity)
print("\n[2] Memisahkan fitur dan target...")
X = df[['loudness']]  # Perhatikan: double bracket untuk 2D array
y = df['popularity']

print(f"Fitur: loudness (kekerasan suara dalam dB)")
print(f"Target: popularity (skor popularitas 0-100)")

# 3. Split data (80% training, 20% testing)
print("\n[3] Split data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Data training: {len(X_train)} lagu")
print(f"Data testing: {len(X_test)} lagu")

# 4. Training model Linear Regression
print("\n[4] Training model Linear Regression...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model selesai dilatih!")

# 5. Tampilkan persamaan linear
print("\n[5] Persamaan Linear yang dihasilkan:")
print(f"Popularity = {model.coef_[0]:.2f} × (loudness) + {model.intercept_:.2f}")
print(f"Atau: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

# 6. Evaluasi model
print("\n[6] Evaluasi model...")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.4f}")

# 7. Visualisasi hasil regresi
print("\n[7] Membuat grafik...")
plt.figure(figsize=(10, 6))

# Scatter plot data aktual
plt.scatter(X, y, color='blue', alpha=0.7, label='Data Aktual')

# Garis regresi
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range = model.predict(X_range)
plt.plot(X_range, y_range, color='red', linewidth=2, label='Garis Regresi')

plt.xlabel('Loudness (dB)', fontsize=12)
plt.ylabel('Popularity Score', fontsize=12)
plt.title('Hubungan Loudness dengan Popularitas Lagu', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regression_plot.png')
plt.show()
print("Grafik disimpan sebagai 'regression_plot.png'")

# 8. Simpan model
print("\n[8] Menyimpan model...")
joblib.dump(model, 'model.pkl')
print("Model disimpan sebagai 'model.pkl'")

# 9. Contoh prediksi
print("\n[9] Contoh prediksi:")
test_loudness = [[-7.0], [-5.0], [-10.0]]
for loud in test_loudness:
    pred = model.predict([loud])[0]
    print(f"  Loudness {loud[0]} dB → Prediksi Popularitas: {pred:.1f}")

print("\n" + "="*60)
print("TRAINING SELESAI!")
print("="*60)