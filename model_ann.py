import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("="*60)
print("TRAINING ARTIFICIAL NEURAL NETWORK (ANN)")
print("Prediksi Popularitas Berdasarkan Loudness")
print("Menggunakan MLPRegressor (Multi-layer Perceptron)")
print("="*60)

print("\n[1] Membaca dataset...")
df = pd.read_csv('data_loudness.csv')
print(f"Data: {len(df)} lagu")
print(df.head())

print("\n[2] Normalisasi data...")
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['loudness', 'popularity']])

X = df_scaled[:, 0].reshape(-1, 1)
Y = df_scaled[:, 1]

print("\n[3] Split data (80% training, 20% testing)...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Training: {len(X_train)} data")
print(f"Testing: {len(X_test)} data")

print("\n[4] Membangun model ANN (MLPRegressor)...")
print("Arsitektur: Input(1) → Hidden(10) → Hidden(10) → Output(1)")
model = MLPRegressor(
    hidden_layer_sizes=(10, 10),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    verbose=False
)

print("\n[5] Training model...")
model.fit(X_train, Y_train)

print("\n[6] Evaluasi model...")
Y_pred = model.predict(X_test)

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

print("\n[7] Visualisasi hasil...")
X_test_original = scaler.inverse_transform(np.column_stack((X_test[:, 0], np.zeros(len(X_test)))))[:, 0]
Y_test_original = scaler.inverse_transform(np.column_stack((np.zeros(len(Y_test)), Y_test)))[:, 1]
Y_pred_original = scaler.inverse_transform(np.column_stack((np.zeros(len(Y_pred)), Y_pred)))[:, 1]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test_original, Y_test_original, color='blue', label='Data Aktual', alpha=0.7)
plt.scatter(X_test_original, Y_pred_original, color='red', label='Prediksi ANN', alpha=0.7)
plt.xlabel('Loudness (dB)')
plt.ylabel('Popularity Score')
plt.title('Prediksi ANN vs Data Aktual')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(Y_test_original, label='Aktual', marker='o')
plt.plot(Y_pred_original, label='Prediksi', marker='x')
plt.xlabel('Data ke-')
plt.ylabel('Popularity Score')
plt.title('Perbandingan Aktual vs Prediksi')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ann_results.png')
plt.show()

print("\n[8] Menyimpan model...")
joblib.dump(model, 'ann_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model disimpan sebagai 'ann_model.pkl' dan 'scaler.pkl'")

print("\n[9] Contoh prediksi:")
test_loudness = np.array([[-7.0], [-5.0], [-10.0], [-3.0], [-12.0]])
test_scaled = scaler.transform(np.column_stack((test_loudness, np.zeros(len(test_loudness)))))[:, 0].reshape(-1, 1)
predictions_scaled = model.predict(test_scaled)
predictions = scaler.inverse_transform(np.column_stack((test_scaled[:, 0], predictions_scaled)))[:, 1]

for loud, pred in zip(test_loudness.flatten(), predictions):
    print(f"Loudness {loud} dB → Prediksi Popularitas: {pred:.1f}")

print("\n" + "="*60)
print("TRAINING ANN SELESAI!")
print("Jalankan 'python app_ann.py' untuk membuka web app")
print("="*60)