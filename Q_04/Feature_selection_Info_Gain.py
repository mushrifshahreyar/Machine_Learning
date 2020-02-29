import Feature_ranking_variance_threshold as q_04
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 
from sklearn import tree
import graphviz

def preprocessing(dataset):
    X = np.delete(dataset,34,axis = 1)
    y = dataset
    for i in range(34):
        y = np.delete(y,0,axis=1)
    
    # y = y.ravel()
    return X,y

def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 

def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred)) 
      
if(__name__ == "__main__"):
    dataset = q_04.read_data()
    X,y = preprocessing(dataset)
    X_train, X_test, y_train, y_test = train_test_split(  
    X, y, test_size = 0.3, random_state = 100) 

    # print(X_test)
    # print(y_test)

    clf_entropy = DecisionTreeClassifier(criterion = 'entropy').fit(X_train,y_train)
    # y_pred_entropy = prediction(X_test, clf_entropy)
    # cal_accuracy(y_test, y_pred_entropy) 
    # print(tree.plot_tree(clf_entropy.fit(X_train,y_train)))

    dot_data = tree.export_graphviz(clf_entropy,out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("image",view=True)