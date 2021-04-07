import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
import graphviz

def read_data():
    dataset = np.genfromtxt('dermatology.data',delimiter=',')
    mean = np.nanmean(dataset,axis=0)
    inds = np.where(np.isnan(dataset))
    dataset[inds] = np.take(mean,inds[1])
    X = np.delete(dataset,34,1)
    y = dataset
    for i in range(34):
        y = np.delete(y,0,1)
    y= np.reshape(y,(1,366))
    y = y.ravel()
    return X,y

def standard_scaler(X_train,X_test):
    scaler = StandardScaler()
    scaler.fit(X)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train,X_test

if(__name__ == "__main__"):
    X,y = read_data()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30)
    print(X_train)
    # X_train,X_test = standard_scaler(X_train,X_test)
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train,y_train)
    y_pred = decision_tree.predict(X_test)
    # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))    
    dot_data = tree.export_graphviz(decision_tree,out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("image",view=True)
    