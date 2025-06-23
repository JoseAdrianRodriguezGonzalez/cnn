from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
import numpy as np

# Cargar datos
train_features = np.load('../db/feature_effnet_s/train_features.npy')
train_labels = np.load('../db/feature_effnet_s/train_labels.npy')
val_features = np.load('../db/feature_effnet_s/val_features.npy')
val_labels = np.load('../db/feature_effnet_s/val_labels.npy')

# Concatenar
X_train_full = np.concatenate([train_features, val_features], axis=0)
y_train_full = np.concatenate([train_labels, val_labels], axis=0)

# Pipeline con orden correcto
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=60)),
    ('svm', OneVsRestClassifier(SVC(kernel='rbf', C=1, gamma='scale')))
])

# Hiperparámetros correctos
param_grid = {
    'svm__estimator__C': loguniform(0.1, 100),
    'svm__estimator__gamma': loguniform(1e-4, 1e-1),
    'pca__n_components': [60],
}

random = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Entrenar (puedes probar con subset primero)
# random.fit(X_train_full, y_train_full)
random.fit(X_train_full[:10000], y_train_full[:10000])  # Subset más rápido

# Resultados
for i, params in enumerate(random.cv_results_['params']):
    mean_score = random.cv_results_['mean_test_score'][i]
    std_score = random.cv_results_['std_test_score'][i]
    print(f"Params: {params} -> Mean Accuracy: {mean_score:.4f} ± {std_score:.4f}")

print("Mejores parámetros:", random.best_params_)
print("Mejor score:", random.best_score_)
##Mejores parámetros: {'pca__n_components': 60, 'svm__estimator__C': np.float64(31.428808908401084), 'svm__estimator__gamma': np.float64(0.0004335281794951569)}
##Mejor score: 0.1373
