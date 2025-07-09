import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import numpy as np

# Cargar datos (sin concatenar)
train_features = np.load('../db/feature_effnet_s/train_features.npy')
train_labels = np.load('../db/feature_effnet_s/train_labels.npy')
val_features = np.load('../db/feature_effnet_s/val_features.npy')
val_labels = np.load('../db/feature_effnet_s/val_labels.npy')

# 1. Verificar etiquetas
print("Etiquetas en train:", np.unique(train_labels))
print("Etiquetas en val:", np.unique(val_labels))

# 2. Escalar correctamente (sin data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_features)  # Solo fit en train
X_val_scaled = scaler.transform(val_features)         # Val se transforma

# 3. Modelo con mayor capacidad y regularización
model = Sequential([
    Dense(512, activation='relu', input_shape=(1280,)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(12, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Early Stopping

# 5. Entrenamiento
history = model.fit(
    train_features,
    train_labels,
    epochs=50,
    batch_size=128,
    validation_data=(val_features, val_labels),
    verbose=1
)

# 6. Evaluación
val_loss, val_acc = model.evaluate(val_features, val_labels, verbose=0)
print(f"\nAccuracy en validación: {val_acc:.4f}")
# sin normalizar Accuracy en validación: 0.2132
#Accuracy en validación: 0.2514