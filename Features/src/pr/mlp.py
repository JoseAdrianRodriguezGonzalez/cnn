import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import hdbscan
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns

# 1. Carga de datos (ya separados en train/val)
train_features = np.load('../../db/feature_effnet_s/train_features.npy')  # (40493, 1200)
train_labels = np.load('../../db/feature_effnet_s/train_labels.npy')      # (40493,)
val_features = np.load('../../db/feature_effnet_s/val_features.npy')      # (10123, 1200)
val_labels = np.load('../../db/feature_effnet_s/val_labels.npy')          # (10123,)

# 2. Reducción de dimensionalidad (SOLO en train)
pca = PCA(n_components=60, random_state=42)
X_train_pca = pca.fit_transform(train_features)  # Ajuste SOLO con train
X_val_pca = pca.transform(val_features)          # Transformar val con mismos parámetros

# 3. Clustering (SOLO en train)
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50,
    min_samples=10,
    metric='euclidean',
    cluster_selection_epsilon=0.5
)
labels_hdb = clusterer.fit_predict(X_train_pca)  # Solo entrenamiento

# 4. Mapeo de clusters a clases (SOLO con train_labels)
cluster_to_class = {}
for cluster_id in np.unique(labels_hdb):
    if cluster_id == -1:  # Ignorar outliers
        continue
    indices = np.where(labels_hdb == cluster_id)[0]
    most_common_class = Counter(train_labels[indices]).most_common(1)[0][0]
    cluster_to_class[cluster_id] = most_common_class
unique_classes = np.unique(list(cluster_to_class.values()))
class_mapping = {old_class: new_class for new_class, old_class in enumerate(unique_classes)}
# Verificación del mapeo
print("\n--- Verificación de mapeo clusters -> clases ---")
for cluster_id, class_id in cluster_to_class.items():
    cluster_mask = (labels_hdb == cluster_id)
    unique, counts = np.unique(train_labels[cluster_mask], return_counts=True)
    print(f"Cluster {cluster_id} -> Clase {class_id}: Distribución real {dict(zip(unique, counts))}")

# 5. Filtrar outliers y preparar datos de entrenamiento (RE-MAPEAR ANTES DEL SPLIT)
train_mask = labels_hdb != -1
X_train_clustered = X_train_pca[train_mask]
y_train_clustered = np.array([class_mapping[cluster_to_class[cl]] for cl in labels_hdb[train_mask]])  # Aplicar mapeo aquí

# Escalado (SOLO ajustado con train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clustered)
X_val_scaled = scaler.transform(X_val_pca)  # Val se escala con parámetros de train

# 6. Split de validación (a partir de train_clustered)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train_clustered, 
    test_size=0.2, 
    stratify=y_train_clustered, 
    random_state=42
)

# 7. Definición del modelo con regularización
# Re-mapear etiquetas
y_train_clustered = np.array([class_mapping[cluster_to_class[cl]] for cl in labels_hdb[train_mask]])
n_classes = len(np.unique(y_train_clustered))  # Usar las clases ya re-mapeadas

print(f"\n--- Entrenando con {n_classes} clases ---")

model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_split.shape[1],), kernel_regularizer='l2'),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer='l2'),
    Dropout(0.3),
    Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 8. Entrenamiento con Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train_split, y_train_split,
    validation_data=(X_val_split, y_val_split),
    epochs=50,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

# 9. Evaluación en val_features original (sin clustering)
# 9. Evaluación en val_features original (sin clustering)
y_val_pred = model.predict(X_val_scaled)  # Obtener predicciones (probabilidades)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)
inverse_class_mapping = {v: k for k, v in class_mapping.items()}
y_val_pred_classes_original = np.array([inverse_class_mapping[pred] for pred in y_val_pred_classes])

# 10. Métricas (usar clases originales)
print("\n--- Resultados en conjunto de validación (clases originales) ---")
print(classification_report(val_labels, y_val_pred_classes_original))
# 11. Visualización de resultados
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_val_pca[:, 0], X_val_pca[:, 1], c=val_labels, cmap='tab20', alpha=0.6)
plt.title("Etiquetas Reales (Val)")

plt.subplot(1, 2, 2)
plt.scatter(X_val_pca[:, 0], X_val_pca[:, 1], c=y_val_pred_classes, cmap='tab20', alpha=0.6)
plt.title("Predicciones del Modelo")
plt.show()