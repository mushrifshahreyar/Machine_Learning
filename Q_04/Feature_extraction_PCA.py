import Feature_ranking_variance_threshold as q_04
import numpy as np
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler

def PCA(dataset):
    standard_scaler = StandardScaler()
    X_standard = standard_scaler.fit_transform(dataset)
    # print(X_standard)
    no_of_components = 2

    while(no_of_components <= 33):
        pca = decomposition.PCA(n_components=no_of_components)
        X_standard_PCA = pca.fit_transform(X_standard)
        var = pca.explained_variance_ratio_
        if(sum(var) >= 0.80):
            break
        no_of_components = no_of_components + 1
    
    # print(X_standard_PCA)
    # print(pca.components_[0])
    
    print(X_standard_PCA.shape)

if(__name__ == "__main__"):
    dataset = q_04.read_data()
    dataset = np.delete(dataset,34,axis=1)
    
    PCA(dataset)