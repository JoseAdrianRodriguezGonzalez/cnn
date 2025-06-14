import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Cargar datos
train_features = np.load('../db/feature_effnet_s/train_features.npy')  # (40493, 1200)
train_labels = np.load('../db/feature_effnet_s/train_labels.npy')      # (40493, 1) o (40493,)
val_features = np.load('../db/feature_effnet_s/val_features.npy')      # (10123, 1200)
val_labels = np.load('../db/feature_effnet_s/val_labels.npy')          # (10123, 1) o (10123,)

# Asegurar que las etiquetas estén en forma 1D
X_all = np.concatenate([train_features, val_features])
y_all = np.concatenate([train_labels, val_labels])
y_all = y_all.ravel()
domain = np.array(['train'] * len(train_labels) + ['val'] * len(val_labels))

X_pca = PCA(n_components=100).fit_transform(X_all)
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_pca)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=domain, style=domain, palette='Set1', s=10)
plt.title("Separación entre train y val con t-SNE")
plt.tight_layout()
plt.show()

domain = np.array(['train'] * len(train_labels) + ['val'] * len(val_labels))
y_all = np.concatenate([train_labels, val_labels])
X_all = np.concatenate([train_features, val_features])

# PCA + t-SNE
X_pca = PCA(n_components=100).fit_transform(X_all)
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_pca)

# Visualización
plt.figure(figsize=(10,7))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y_all, style=domain, palette='tab10', s=10)
plt.title("Distribución por clase y conjunto (train/val)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(train_features)
X_val_pca = pca.transform(val_features)

svm = SVC(kernel='rbf', C=10)
svm.fit(X_train_pca, train_labels)

y_pred = svm.predict(X_val_pca)
print(accuracy_score(val_labels, y_pred))
