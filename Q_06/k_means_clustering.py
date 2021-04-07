from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score,v_measure_score
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random

def read_data():
    
    main_string = '20_newsgroups/'
    sub_string = ["comp.graphics/3793",
                  "comp.sys.ibm.pc.hardware/5883",
                  "rec.autos/10156",
                  "rec.motorcycles/10312",
                  "talk.politics.guns/5330"
                  ]
    data = []
    for sub in sub_string:
        string = main_string + sub
        for i in range(10):
            string_n = string + str(i)
            with open(string_n,"r",errors="ignore") as f:
                for p in range(11):
                    next(f)
                data.append(f.read())
    
    test_data = []
    
    strings = ['comp.graphics/3794',
               'comp.sys.ibm.pc.hardware/5890',
               'rec.autos/10157',
               'comp.graphics/3795',
               'rec.motorcycles/10313',
               'talk.politics.guns/5333']
    
    rand = random.randint(2,9)
    for sub in strings:
        string = main_string + sub + str(rand)
        with open(string,"r",errors="ignore") as f:
            for p in range(11):
                    next(f)
            test_data.append(f.read())
    
    return data,test_data

def text_preprocessing(documents):
    
    for i in range(len(documents)):
        documents[i] = documents[i].strip()
        documents[i] = documents[i].lower()

        documents[i] = re.sub("\d+","",documents[i])
        documents[i] = re.sub("[^a-zA-Z]+"," ",documents[i])


def tokenizing(text_array):
    for i in range(len(text_array)):
        text_array = word_tokenize(text_array)

    lemmetizer = WordNetLemmatizer()
    text_array_new = text_array
    for i in range(len(text_array)):
        for j in range(len(text_array[i])):
            text_array_new[i][j] = lemmetizer.lemmatize(text_array_new[i][j])
    
    return text_array_new

if(__name__ == "__main__"):
    documents, targets = read_data()
    text_preprocessing(documents)

    # tokenizing(documents)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    y = [0,1,2,0,3,4]
    true_k = 5
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)

    model.fit(X)
    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind]),
        print
    
    targets_tr = vectorizer.transform(targets)
    y_predict = model.predict(targets_tr)

    print(homogeneity_score(y,y_predict))
    print(v_measure_score(y,y_predict))