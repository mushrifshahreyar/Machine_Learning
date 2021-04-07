import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

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
    X_train,X_test = standard_scaler(X_train,X_test)
    
    logisticRegression = LogisticRegression(solver='liblinear',penalty='l1')
    # logisticRegression = LogisticRegression(solver='lbfgs',penalty='l2')
    
    logisticRegression.fit(X_train,y_train)
    y_pred = logisticRegression.predict(X_test)
    count_misclassified = (y_test != y_pred).sum()
    print('Misclassified samples: {}'.format(count_misclassified))
    accuracy = metrics.accuracy_score(y_test,y_pred)
    print(accuracy)
