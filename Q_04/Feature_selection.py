import Feature_ranking_variance_threshold as q_04
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt

def preprocessing(dataset):
    X = np.delete(dataset,34,axis = 1)
    y = dataset
    for i in range(34):
        y = np.delete(y,0,axis=1)
    # print(y.shape)
    y = y.ravel()
    return X,y

def Feature_forward(X,y):
    randomeforestclassifier = RandomForestClassifier(n_estimators=50, criterion='entropy',random_state=4)
    sfs = SFS(randomeforestclassifier,k_features=10,forward=True,floating=False,verbose=2,scoring='accuracy',cv=5,n_jobs=-1)
    # print(y)
    sfs.fit(X,y)
    print(sfs.k_feature_names_)
    fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
    plt.title('Sequential Forward Selection (w. StdErr)')
    plt.grid()
    plt.show()
    

def Feature_backward(X,y):
    randomeforestclassifier = RandomForestClassifier(n_estimators=50, criterion='entropy',random_state=4)
    sfs = SFS(randomeforestclassifier,k_features=(1,33),forward=False,floating=False,verbose=2,scoring='accuracy',cv=5,n_jobs=-1)
    # print(y)
    sfs.fit(X,y)
    print(sfs.k_feature_names_)

if(__name__ == "__main__"):
    dataset = q_04.read_data()
    X,y = preprocessing(dataset)
    Feature_forward(X,y)
    