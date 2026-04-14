# train_ann.py - Menggunakan ANN 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("="*60)
print("TRAINING ARTIFICIAL NEURAL NETWORK (ANN)")
print("Prediksi Popularitas Berdasarkan Loudness")
print("="*60)

#  dataset
print("\n[1] Membaca dataset...")
df = pd.read_csv('data_loudness.csv')
print(f"Data: {len(df)} lagu")
print(df.head())

print("\n[2] Normalisasi data...")
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['loudness', 'popularity']])

X = df_scaled[:, 0].reshape(-1, 1)  # loudness sebagai input
Y = df_scaled[:, 1]                   # popularity sebagai output

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Training: {len(X_train)} data")
print(f"Testing: {len(X_test)} data")

print("\n[3] Membangun model ANN...")
model = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),  # Hidden layer 1 (10 neuron)
    Dense(10, activation='relu'),                     # Hidden layer 2 (10 neuron)
    Dense(1, activation='linear')                     # Output layer
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(model.summary())

print("\n[4] Training model ANN...")
history = model.fit(
    X_train, Y_train,
    epochs=200,
    validation_data=(X_test, Y_test),
    verbose=1
)

print("\n[5] Evaluasi model...")
loss, mae = model.evaluate(X_test, Y_test)
print(f"Loss (MSE): {loss:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss selama Training')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.title('MAE selama Training')

plt.tight_layout()
plt.savefig('ann_training_history.png')
plt.show()

Y_pred = model.predict(X_test)

# Balikkan skala ke nilai asli
X_test_original = scaler.inverse_transform(np.column_stack((X_test[:, 0], np.zeros(len(X_test)))))[:, 0]
Y_test_original = scaler.inverse_transform(np.column_stack((np.zeros(len(Y_test)), Y_test)))[:, 1]
Y_pred_original = scaler.inverse_transform(np.column_stack((np.zeros(len(Y_pred)), Y_pred.flatten())))[:, 1]

plt.figure(figsize=(10, 6))
plt.scatter(X_test_original, Y_test_original, color='blue', label='Data Aktual', alpha=0.7)
plt.scatter(X_test_original, Y_pred_original, color='red', label='Prediksi ANN', alpha=0.7)
plt.xlabel('Loudness (dB)')
plt.ylabel('Popularity Score')
plt.title('Hasil Prediksi ANN vs Data Aktual')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('ann_predictions.png')
plt.show()

print("\n[6] Menyimpan model...")
model.save('ann_model.h5')
import joblib
joblib.dump(scaler, 'scaler.pkl')
print("Model disimpan sebagai 'ann_model.h5' dan 'scaler.pkl'")

print("\n[7] Contoh prediksi:")
test_loudness = np.array([[-7.0], [-5.0], [-10.0]])
test_scaled = scaler.transform(np.column_stack((test_loudness, np.zeros(len(test_loudness)))))[:, 0].reshape(-1, 1)
predictions_scaled = model.predict(test_scaled)
predictions = scaler.inverse_transform(np.column_stack((test_scaled[:, 0], predictions_scaled[:, 0])))[:, 1]

for loud, pred in zip([-7.0, -5.0, -10.0], predictions):
    print(f"Loudness {loud} dB → Prediksi Popularitas: {pred:.1f}")

print("\n" + "="*60)
print("TRAINING ANN SELESAI!")
print("Jalankan 'python app_ann.py' untuk web app")
print("="*60)