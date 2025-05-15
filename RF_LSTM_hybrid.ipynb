# Cell 1: Imports and Data Definition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Sample dataset
data = {
    "Time": [
        "07:00 AM","08:00 AM","09:00 AM","10:00 AM","11:00 AM","12:00 PM",
        "01:00 PM","02:00 PM","03:00 PM","04:00 PM","05:00 PM","06:00 PM",
        "07:00 PM","08:00 PM"
    ],
    "Vehicle Count": [300,450,600,700,800,750,650,500,550,650,700,800,700,600],
    "Flow": [5.0,7.5,10.0,11.7,13.3,12.5,10.8,8.3,9.2,10.8,11.7,13.3,11.7,10.0],
    "Signal Time": [35,40,45,45,50,50,45,40,40,45,50,50,45,40],
    "Avg Speed": [35,30,25,20,20,22,25,30,28,25,20,18,22,30],
    "Accident": [0,0,1,0,0,0,0,0,0,0,0,0,1,0]
}
df = pd.DataFrame(data)

# Encode Time to pandas datetime if needed (optional)
# df['Time'] = pd.to_datetime(df['Time'], format='%I:%M %p')

# Cell 2: Random Forest Feature Importance
# Prepare features and target (next Vehicle Count)
df['Target'] = df['Vehicle Count'].shift(-1)
df_rf = df.dropna().drop(columns=['Time'])
X = df_rf.drop(columns=['Target'])
y = df_rf['Target']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("Feature importances:\n", importances)

# Select top 4 features
top_features = importances.index[:4].tolist()
print("\nTop features:", top_features)

# Cell 3: Prepare Sequences for LSTM
arr = df_rf[top_features].values
time_steps = 3
X_seq, y_seq = [], []
for i in range(len(arr) - time_steps):
    X_seq.append(arr[i:i+time_steps])
    y_seq.append(arr[i+time_steps][top_features.index('Vehicle Count')])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# Train/validation split
split = int(0.8 * len(X_seq))
X_train, X_val = X_seq[:split], X_seq[split:]
y_train, y_val = y_seq[:split], y_seq[split:]

# Cell 4: Build & Train LSTM
model = Sequential([
    LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=1,
    validation_data=(X_val, y_val),
    verbose=1
)

# Cell 5: Plot Training History
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('LSTM Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()
