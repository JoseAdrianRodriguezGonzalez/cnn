import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data=np.load('./modelo/features_val.npz')
print("Keys: ",data.files)
data_dict=dict(data);
#print("X: ",data_dict['X'])
#print("Y: ",data_dict['y'])
x=data_dict['X']
y=data_dict['y']
#print(np.unique(y)) #These are the classes
#print(y.shape)
#print(x.shape)
#Arbol
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=42)
forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=2,class_weight='balanced')
forest.fit(X_train,y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1][:500]
print(f"Indices importantes: {indices}")
print(f"Accuracy on train: {forest.score(X_train,y_train)}")
print(f"Accuracy on test: {forest.score(X_test,y_test)}")
#Con los principañles
X_train,X_test,y_train,y_test=train_test_split(x[:,indices],y,stratify=y,random_state=42)
forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=2,class_weight='balanced')
forest.fit(X_train,y_train)
print("Despues de filtrar")
print(f"Accuracy on train: {forest.score(X_train,y_train)}")
print(f"Accuracy on test: {forest.score(X_test,y_test)}")

##Con PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=100)
X_pca = pca.fit_transform(x)

forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=2, class_weight='balanced')
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, stratify=y, random_state=42)
forest.fit(X_train, y_train)

print(f"Train acc: {forest.score(X_train, y_train)}")
print(f"Test acc: {forest.score(X_test, y_test)}")

###Let me validate it

from sklearn.manifold import TSNE

X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_pca)
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette='tab10', s=10)
plt.title("Distribución de clases con t-SNE")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()



#Cross validation
from sklearn.model_selection import cross_val_score
forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=2, class_weight='balanced')
scores=cross_val_score(forest,x,y,cv=5,scoring='accuracy')
print(f"Scores with cross validation: {scores}")

###################


import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Suponiendo que ya tienes tus datos cargados en X (features) e y (etiquetas)
# X = ...
# y = ...

# División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, stratify=y, random_state=42
)

# Definir el clasificador base con GPU
xgb_clf = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(np.unique(y)),
    use_label_encoder=False,
    eval_metric="mlogloss",
    tree_method="hist", 
    device="cuda",        # << GPU
    predictor="predictor",      # << GPU
    random_state=42
)

# Espacio de búsqueda de hiperparámetros
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.3, 0.5],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [1, 1.5, 2.0]
}

# Búsqueda aleatoria con validación cruzada
random_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_dist,  # <- Este nombre es clave
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Entrenamiento del modelo optimizado
random_search.fit(X_train, y_train)

# Mejor modelo
best_model = random_search.best_estimator_
print("Mejores hiperparámetros encontrados:")
print(random_search.best_params_)

# Predicción y evaluación
y_pred = best_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))