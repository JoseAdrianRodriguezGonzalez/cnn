from sklearn.preprocessing import StandardScaler
import numpy as np
#All of them are numpy arrays
train_features=np.load('../db/feature_effnet_s/train_features.npy') #Matriz of 40493 x 1200
train_labels=np.load('../db/feature_effnet_s/train_labels.npy') #Values between 0 to 11 40493 x 1
val_features=np.load('../db/feature_effnet_s/val_features.npy') #10123 x 1200 
val_labels=np.load('../db/feature_effnet_s/val_labels.npy') #Values between 0 to 11 10123 x1 

#Escala
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_features)
X_val_scaled = scaler.transform(val_features)

from sklearn.decomposition import PCA

pca = PCA(n_components=100)  # Empieza con 100 componentes (ajusta según resultados)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Modelo inicial (k=5 por defecto)
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_pca, train_labels)

# Predicciones
val_preds = knn.predict(X_val_pca)
accuracy = accuracy_score(val_labels, val_preds)
print(f"Accuracy inicial: {accuracy:.4f}")



from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_neighbors': [3, 5, 7, 10, 15, 20],
    'metric': ['euclidean', 'manhattan', 'cosine'],
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_pca, train_labels)

print("Mejores parámetros:", grid_search.best_params_)
print("Mejor accuracy (validación cruzada):", grid_search.best_score_)

# Evaluación en val_features
best_knn = grid_search.best_estimator_
val_preds = best_knn.predict(X_val_pca)
print("Accuracy en val:", accuracy_score(val_labels, val_preds))


