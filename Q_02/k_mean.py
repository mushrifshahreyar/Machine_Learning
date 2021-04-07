import numpy as np
import math
from matplotlib import pyplot as plt

def read_data(dataname):
    datas = np.array([[0,0,0,0,0]])
    # target = np.array([[]])
    if(dataname == "Iris"):
        with open("Iris/iris.data") as f:
            for line in f:
                data = line.strip().split(',')
                if(len(data) == 5):
                    temp = np.array(data[0:4])
                    temp = temp.astype(np.float)
                    if(data[4] == "Iris-setosa"):
                        temp = np.append(temp,[int(0)])
                    elif(data[4] == "Iris-versicolor"):
                        temp = np.append(temp,[int(1)])
                    else:
                        temp = np.append(temp,[int(2)])
                    datas = np.vstack([datas,temp])
                    
    datas = np.delete(datas,(0),axis=0)
    return datas

def redifine_cluster(data_set,cluster,cluster_no=3):
    distance = np.zeros((150,cluster_no))
    for i in range(cluster_no):
        for j in range(len(data_set[:,1])):
            distance[j][i] = math.sqrt(((data_set[j][0] - cluster[i][0]) ** 2) + ((data_set[j][1] - cluster[i][1]) ** 2) + ((data_set[j][2] - cluster[i][2]) ** 2) + ((data_set[j][3] - cluster[i][3]) ** 2))         

    new_cluster = np.argmin(distance,axis = 1)

    for i in range(len(new_cluster)):
        data_set[i][4] = new_cluster[i]
    
    # return data_set    
    
def k_mean_cluster(data_set,cluster_no = 3):
    cluster = []
    for i in range(cluster_no):
        mean = [0,0,0,0]
        length = 0
        for j in range(len(data_set[:,1])):
            if(data_set[j][4] == i):
                length = length + 1
                mean[0] = mean[0] + data_set[j][0]
                mean[1] = mean[1] + data_set[j][1]
                mean[2] = mean[2] + data_set[j][2]
                mean[3] = mean[3] + data_set[j][3]
        if(length != 0):
            mean[:] = [x / length for x in mean]

        cluster.append(mean)
    # print(cluscluster_mean = np.ones((3,4))ter)
    return cluster


def wcc(dataset,cluster,cluster_no = 3):
    # print(dataset)
    
    distance = np.zeros((150,1))
    for i in range(cluster_no):

        for j in range(len(data_set[:,1])):
            if(data_set[j][4] == i):
                distance[j] = ((data_set[j][0] - cluster[i][0]) ** 2) + ((data_set[j][1] - cluster[i][1]) ** 2) + ((data_set[j][2] - cluster[i][2]) ** 2) + ((data_set[j][3] - cluster[i][3]) ** 2)

    sum = 0
    for j in range(len(distance)):
        sum = sum + distance[j]
    print(sum)
    return sum

def get_random_cluster(data_set,cluster_no = 3):
    cluster = np.array([0,0,0,0])
    for p in range(cluster_no):
        i = np.random.randint(low=5,high=140)
        cluster = np.vstack([cluster,data_set[i,0:4]])

    cluster = np.delete(cluster,(0),axis=0)
    return cluster

def match_array(old,new_arr,cluster_no = 3):
    result = True
    for i in range(3):
        if(old[i][0] == new_arr[i][0] and old[i][1] == new_arr[i][1] and old[i][2] == new_arr[i][2] and old[i][3] == new_arr[i][3]):
            result = True
        else:
            result = False

        if(result == False):
            break

    return result

if(__name__ == "__main__"):
    data_set = read_data("Iris")
    
    # print(data_set)
    
    cluster_mean = get_random_cluster(data_set)
    
    wcc_data = []
    len_i = 0
    while(len_i <= 5):
        redifine_cluster(data_set,cluster_mean,len_i)
        new_cluster_mean = k_mean_cluster(data_set)
        # print(cluster_mean)
        len_i = len_i + 1

        val = wcc(data_set,cluster_mean)
        wcc_data.append(val)

        cluster_mean = new_cluster_mean   
        
    epoch_arr = [x for x in range(len_i)]

    plt.plot(epoch_arr,wcc_data) 
    plt.xlabel("k")
    plt.ylabel("wcc")
    # print(theta)
    plt.show()   
    # # print(data_set)
    
    


