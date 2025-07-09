import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from collections import Counter
import umap
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Carga de datos
train_features = np.load('../../db/feature_effnet_s/train_features.npy')
train_labels = np.load('../../db/feature_effnet_s/train_labels.npy')
val_features = np.load('../../db/feature_effnet_s/val_features.npy')
val_labels = np.load('../../db/feature_effnet_s/val_labels.npy')

# 2. Reducción de dimensionalidad (UMAP SOLO en train)
umap_model = umap.UMAP(n_components=60, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
X_train_umap = umap_model.fit_transform(train_features)  # Ajuste SOLO con train
X_val_umap = umap_model.transform(val_features)          # Transformar val con mismos parámetros

# 3. Clustering con KMeans
n_clusters = 12  # Puedes ajustar este valor
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels_kmeans = kmeans.fit_predict(X_train_umap)

# 4. Mapeo de clusters a clases verdaderas
cluster_to_class = {}
for cluster_id in np.unique(labels_kmeans):
    indices = np.where(labels_kmeans == cluster_id)[0]
    most_common_class = Counter(train_labels[indices]).most_common(1)[0][0]
    cluster_to_class[cluster_id] = most_common_class

unique_classes = np.unique(list(cluster_to_class.values()))
class_mapping = {old_class: new_class for new_class, old_class in enumerate(unique_classes)}

print("\n--- Verificación de mapeo clusters -> clases ---")
for cluster_id, class_id in cluster_to_class.items():
    cluster_mask = (labels_kmeans == cluster_id)
    unique, counts = np.unique(train_labels[cluster_mask], return_counts=True)
    print(f"Cluster {cluster_id} -> Clase {class_id}: Distribución real {dict(zip(unique, counts))}")

# 5. Preparar datos de entrenamiento con etiquetas mapeadas
X_train_clustered = X_train_umap
y_train_clustered = np.array([class_mapping[cluster_to_class[cl]] for cl in labels_kmeans])

# 6. Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clustered)
X_val_scaled = scaler.transform(X_val_umap)

# 7. División de entrenamiento/validación
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train_clustered,
    test_size=0.2,
    stratify=y_train_clustered,
    random_state=42
)

# 8. Modelo (MLP)
n_classes = len(np.unique(y_train_clustered))
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

# 9. Entrenamiento
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train_split, y_train_split,
    validation_data=(X_val_split, y_val_split),
    epochs=50,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

# 10. Evaluación final
y_val_pred = model.predict(X_val_scaled)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)

# Deshacer mapeo
inverse_class_mapping = {v: k for k, v in class_mapping.items()}
y_val_pred_classes_original = np.array([inverse_class_mapping[pred] for pred in y_val_pred_classes])

print("\n--- Resultados en conjunto de validación (clases originales) ---")
print(classification_report(val_labels, y_val_pred_classes_original))
# Obtener clusters en validación
val_clusters = kmeans.predict(X_val_umap)

# Mapear clusters a etiquetas clusterizadas (las mismas clases que el modelo)
y_val_clustered = np.array([class_mapping[cluster_to_class[cl]] for cl in val_clusters])

# Evaluar con datos transformados y etiquetas clusterizadas
val_loss, val_acc = model.evaluate(X_val_scaled, y_val_clustered, verbose=0)
print(f"Accuracy con model.evaluate: {val_acc:.4f}")

# 11. Visualización
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_val_umap[:, 0], X_val_umap[:, 1], c=val_labels, cmap='tab20', alpha=0.6)
plt.title("Etiquetas Reales (Val)")

plt.subplot(1, 2, 2)
plt.scatter(X_val_umap[:, 0], X_val_umap[:, 1], c=y_val_pred_classes, cmap='tab20', alpha=0.6)
plt.title("Predicciones del Modelo")
plt.show()
