import numpy as np
#All of them are numpy arrays
train_features=np.load('../db/feature_effnet_s/train_features.npy') #Matriz of 40493 x 1200
train_labels=np.load('../db/feature_effnet_s/train_labels.npy') #Values between 0 to 11 40493 x 1
val_features=np.load('../db/feature_effnet_s/val_features.npy') #10123 x 1200 
val_labels=np.load('../db/feature_effnet_s/val_labels.npy') #Values between 0 to 11 10123 x1 
print(val_labels.shape) 
#In a moment I will train them


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=300, max_depth=30, random_state=2,class_weight=None)
forest.fit(train_features,train_labels)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1][:500]
print(f"Indices importantes: {indices}")
print(f"Accuracy on train: {forest.score(train_features,train_labels)}")
print(f"Accuracy on test: {forest.score(val_features,val_labels)}")

#from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 20, 30],
#     'class_weight': ['balanced', None]
# }

# search = GridSearchCV(RandomForestClassifier(random_state=2), param_grid, cv=3, scoring='accuracy')
# search.fit(train_features, train_labels.ravel())

# print(f"Best parameters: {search.best_params_}")
# print(f"Best cross-val accuracy: {search.best_score_:.3f}")
##
#$Best parameters: {'class_weight': None, 'max_depth': 30, 'n_estimators': 300}
#Best cross-val accuracy: 0.176
