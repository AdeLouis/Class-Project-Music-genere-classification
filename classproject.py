#Name : Louis Gomez
#       Machine Learning Class Project
#       Python script for the containing associated code

import pandas as pd
import numpy as np
#import spotipy
#import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def knnmodel(X_train,y_train,X_test):
    """This function performs traning and testing evaluation
        using K nearest neighbors"""
    neigh = KNeighborsClassifier(n_neighbors = grid.best_estimator_.n_neighbors,n_jobs = -1)
    neigh.fit(X_train,y_train)
    predictions = neigh.predict(X_test)

    return predictions


def rfmodel(X_train,y_train,X_test):
    """This function performs training and testing evaluation
        using a random foreat model"""
    num = rf_random.best_params_.n_estimators
    rf = RandomForestClassifier(n_estimators = num,max_depth = 5,min_samples_split=5,
                                min_samples_leaf=2,random_state=42)
    rf.fit(X_train,y_train)
    predictions = rf.predict(X_test)

    return predictions


def make_confusion_plots(y_test,y_pred):
    """This functionis used to create the confusion matrices from the testing evaluation"""
    #Covert the class labels back to their categorical form
    y_test = list(label_encoder.inverse_transform(y_test))
    y_predict = list(label_encoder.inverse_transform(y_pred))
    
    
    data = confusion_matrix(y_test, y_predict)
    df_cm = pd.DataFrame(data, columns=np.unique(y_test), index = np.unique(y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    plt.title("Confusion Matrix")
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
    plt.savefig("confusion.png",transparent = True)

    
def gridsearch_knn(X_train,y_train,num_cv):
    """This function passes paramters to the actual gridsearch function"""

  k_range = list(range(1,31))
  param_grid = dict(n_neighbors = k_range)
  knn = KNeighborsClassifier()
  global grid
  grid = GridSearchCV(knn, param_grid, cv = num_cv, scoring = "accuracy", n_jobs = -1,verbose = False)
  grid.fit(X_train, y_train)

  print(grid.best_estimator_)
  print("Best score is: " + str(grid.best_score_))

    
def randomgridsearch_rf(X_train,y_train):
    """Used to randomgridsearch for the random forest due to exponential growth of 
        computation time."""
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 200, num = 5)]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 50, num = 10)]
    max_depth.append(None)
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth}
    global rf_random
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                                    n_iter = 100, cv = 3, verbose=False, random_state=42, n_jobs = -1)
    rf_random.fit(X_train,y_train)

    
def main():
  #Load the dataset
  data = pd.read_csv("/musicdata.csv",index_col = 0)
  cols = data.columns
  labels = data.Genre.values

  x_data = data.drop(columns = "Genre")
  scaler = StandardScaler()
  #Scale each feature with a mean on zero, variance of 1    
  x_new = scaler.fit_transform(x_data)
  
  #Converts categorical labels into numeric labels
  global label_encoder
  label_encoder = LabelEncoder()
  y_labels = label_encoder.fit_transform(labels)
  
  #Divide the dataset into training and testing data with 30% being used for testing    
  X_train, X_test, y_train, y_test = train_test_split(x_new, y_labels, test_size = 0.3,random_state = 42)
    
  #Perform a search over 10 splits over a range of K
  gridsearch_knn(X_train,y_train,10)
  y_pred_knn = knnmodel(X_train,y_train,X_test)
  print(classification_report(y_test,y_pred_knn,target_names = ["Country", "EDM", "RnB", "Rap"]))
  #make_confusion_plots(y_test,y_pred_knn)

  randomgridsearch_rf(X_train,y_train)
  y_pred_rf = rfmodel(X_train,y_train,X_test)
  print(classification_report(y_test,y_pred_rf,target_names = ["Country", "EDM", "RnB", "Rap"]))
  #make_confusion_plots(y_test,y_pred_knn)

if __name__ == "__main__":
    main()
