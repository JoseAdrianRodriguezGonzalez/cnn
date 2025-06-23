import time
from sklearn.neural_network import MLPClassifier
import numpy as np

# Carga de datos
train_features = np.load('../db/feature_effnet_s/train_features.npy')
train_labels = np.load('../db/feature_effnet_s/train_labels.npy')
val_features = np.load('../db/feature_effnet_s/val_features.npy')
val_labels = np.load('../db/feature_effnet_s/val_labels.npy')

# Concatenación
X_train_full = np.concatenate([train_features, val_features], axis=0)
y_train_full = np.concatenate([train_labels, val_labels], axis=0)

# Clasificador MLP
clf = MLPClassifier(hidden_layer_sizes=(256,), activation='relu', max_iter=300, random_state=42)

# Tiempo de entrenamiento
start_time = time.time()
clf.fit(X_train_full, y_train_full)
end_time = time.time()

# Resultados
print("Accuracy:", clf.score(val_features, val_labels))
print(f"Tiempo de entrenamiento: {end_time - start_time:.2f} segundos")
