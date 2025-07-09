import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import hdbscan


train_features = np.load('../../db/feature_effnet_s/train_features.npy')  # (40493, 1200)
train_labels = np.load('../../db/feature_effnet_s/train_labels.npy')      # (40493,)
val_features = np.load('../../db/feature_effnet_s/val_features.npy')      # (10123, 1200)
val_labels = np.load('../../db/feature_effnet_s/val_labels.npy')          # (10123,)##

##Prueba con la visualización por clase

X = np.vstack([train_features, val_features])
y = np.concatenate([train_labels, val_labels])
###PCA 60 COMPONENTES
pca = PCA(n_components=60,random_state=42)
X_pca = pca.fit_transform(X)
def figures_plotting(data, labels, s_value=5, a=0.7, title='', cmap_val='tab20'):
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)

    if np.issubdtype(labels.dtype, np.integer):
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        palette = sns.color_palette(cmap_val, n_colors=max(n_clusters, 1))

        color_dict = {}
        idx = 0
        for label in unique_labels:
            if label == -1:
                color_dict[label] = (0.5, 0.5, 0.5)  # Gris para outliers
            else:
                if idx >= len(palette):
                    palette.extend(sns.color_palette(cmap_val, n_colors=10))  # Reusar colores
                color_dict[label] = palette[idx]
                idx += 1

        colors = [color_dict[label] for label in labels]

        # Dibuja cada grupo por separado para permitir la leyenda
        for label in unique_labels:
            mask = labels == label
            label_name = f'Outlier' if label == -1 else f'Cluster {label}'
            plt.scatter(data[mask, 0], data[mask, 1], c=[color_dict[label]], label=label_name,
                        s=s_value, alpha=a)
        plt.legend(markerscale=2, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(data[:, 0], data[:, 1], c=labels, s=s_value, alpha=a)

    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.show()

#Utilizar hdbscan como alterntaiva

clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10, metric='euclidean',cluster_selection_epsilon=0.5)
labels_hdb = clusterer.fit_predict(X_pca)

#etiqeutar por cluster las classes
from collections import Counter
cluster_to_class={}
for id in np.unique(labels_hdb):
    if id==-1:
        continue
    indices=np.where(labels_hdb==id)[0]
    clase_pred=Counter(y[indices]).most_common(1)[0][0]
    cluster_to_class[id]=clase_pred
#Asignar clases segun cluster

y_clustered=np.full_like(labels_hdb,fill_value=-1)
for idx, cl in enumerate(labels_hdb):
    if cl in cluster_to_class:
        y_clustered[idx]=cluster_to_class[cl]

from sklearn.ensemble import RandomForestClassifier

# Solo usar puntos con etiquetas válidas (ignorar outliers)
mask = y_clustered != -1
X_train_clustered = X_pca[mask]
y_train_clustered = y_clustered[mask]

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_clustered, y_train_clustered)


#Let´s change everything
# Transforma val_features con el mismo PCA entrenado
val_pca = pca.transform(val_features)

# Predice clases con el clasificador entrenado
y_val_pred = clf.predict(val_pca)

# Compara contra las etiquetas verdaderas
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(val_labels, y_val_pred))

print(f"Total de datos: {len(X)}")
print(f"Etiquetados por cluster: {np.sum(y_clustered != -1)}")
