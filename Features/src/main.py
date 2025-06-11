import numpy as np
#All of them are numpy arrays
train_features=np.load('../db/feature_effnet_s/train_features.npy') #Matriz of 40493 x 1200
train_labels=np.load('../db/feature_effnet_s/train_labels.npy') #Values between 0 to 11 40493 x 1
val_features=np.load('../db/feature_effnet_s/val_features.npy') #10123 x 1200 
val_labels=np.load('../db/feature_effnet_s/val_labels.npy') #Values between 0 to 11 10123 x1 
print(val_labels.shape) 
#In a moment I will train them