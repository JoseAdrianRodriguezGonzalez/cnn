import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import hdbscan
import umap

# Cargar datos
train_features = np.load('../db/feature_effnet_s/train_features.npy')  # (40493, 1200)
train_labels = np.load('../db/feature_effnet_s/train_labels.npy')      # (40493,)
val_features = np.load('../db/feature_effnet_s/val_features.npy')      # (10123, 1200)
val_labels = np.load('../db/feature_effnet_s/val_labels.npy')          # (10123,)##

##Prueba con la visualización por clase

X = np.vstack([train_features, val_features])
y = np.concatenate([train_labels, val_labels])
###PCA 60 COMPONENTES
pca = PCA(n_components=60)
X_pca = pca.fit_transform(X)

# Aplicar t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42,init='pca',learning_rate='auto')
X_tsne = tsne.fit_transform(X_pca)

# Visualizar por clase
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab20', s=2, alpha=0.7)
plt.colorbar(scatter, label='Etiqueta de clase')
plt.title('Distribución por clase con t-SNE sobre PCA (60 comp.)')
plt.grid()
plt.tight_layout()
plt.show()

#APLCIAR KMEANS PARA VIOSULAIZAR LOS CLUSTERES
kmeans=KMeans(n_clusters=12,random_state=42)
clusters=kmeans.fit_predict(X_tsne)
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='tab10', s=2, alpha=0.7)
plt.title('Clusters con K-Means sobre t-sne')
plt.grid()
plt.show()
#Utilizar hdbscan como alterntaiva

clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10, metric='euclidean')
labels_hdb = clusterer.fit_predict(X_tsne)

# Número de clusters (excluyendo outliers etiquetados como -1)
n_clusters = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
print(f"Clusters encontrados (sin contar outliers): {n_clusters}")

# Visualización
plt.figure(figsize=(10, 8))
palette = sns.color_palette('tab20', n_colors=len(set(labels_hdb)))
colors = [palette[label] if label >= 0 else (0.5, 0.5, 0.5) for label in labels_hdb]

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, s=5, alpha=0.7)
plt.title(f'HDBSCAN sobre UMAP - {n_clusters} clusters')
plt.grid(True)
plt.tight_layout()
plt.show()




###########################################
#umap

reducer = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_pca)

# Visualización coloreando por clase
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab20', s=5, alpha=0.6)
plt.colorbar(scatter, label='Clase')
plt.title('UMAP sobre PCA (60 comp.)')
plt.grid(True)
plt.show()

#Utilizar kmeans 

kmeans=KMeans(n_clusters=2,random_state=42)
clusters=kmeans.fit_predict(X_umap)
plt.figure(figsize=(10, 8))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=clusters, cmap='tab10', s=2, alpha=0.7)
plt.title('Clusters con K-Means sobre t-sne')
plt.grid()
plt.show()

####Utilizar hdbscan para umap
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10, metric='euclidean')
labels_hdb = clusterer.fit_predict(X_umap)

# Número de clusters (excluyendo outliers etiquetados como -1)
n_clusters = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
print(f"Clusters encontrados (sin contar outliers): {n_clusters}")

# Visualización
plt.figure(figsize=(10, 8))
palette = sns.color_palette('tab20', n_colors=len(set(labels_hdb)))
colors = [palette[label] if label >= 0 else (0.5, 0.5, 0.5) for label in labels_hdb]

plt.scatter(X_umap[:, 0], X_umap[:, 1], c=colors, s=5, alpha=0.7)
plt.title(f'HDBSCAN sobre UMAP - {n_clusters} clusters')
plt.grid(True)
plt.tight_layout()
plt.show()

 # Clases en el gran cluster

#########################################
####Encontrar la cantidad minima de componente wque superen el 90%
# PCA sobre las características de entrenamiento
pca = PCA(n_components=100)
X_pca = pca.fit_transform(train_features)  # <- Aquí se ajusta el PCA

# Gráfica de varianza acumulada
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Número de componentes')
plt.ylabel('Varianza acumulada')
plt.title('Varianza explicada por PCA')
plt.axhline(y=0.9, color='r', linestyle='--', label='90% varianza')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
###################################
#Con esta prueba pude determinar que con tan solo 60 componentes ya expplcia todo

pca_60=PCA(n_components=60)
X_train_pca=pca_60.fit_transform(train_features)
X_val_pca=pca_60.fit_transform(val_features)
X_all_pca = np.concatenate([X_train_pca, X_val_pca])
domain = np.array(['train'] * len(train_labels) + ['val'] * len(val_labels))

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_all_pca)

# Gráfica
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=domain, palette=["red", "blue"], s=10)
plt.title('Separación entre train y val con t-SNE sobre PCA (60 comp.)')
plt.grid(True)
plt.show()
#####################################################


