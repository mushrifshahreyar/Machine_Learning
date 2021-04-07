import numpy as np
import math
from matplotlib import pyplot as plt

def read_data(dataname):
    datas = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0]])
    # target = np.array([[]])
    if(dataname == "Vowel"):
        with open("Vowels/ae.train") as f:
            for line in f:
                data = line.split()
                
                if(len(data) == 12):
                    temp = np.array(data[0:13])
                    temp = temp.astype(np.float)
                    temp = np.append(temp,[int(0)])
                    datas = np.vstack([datas,temp])
                
    datas = np.delete(datas,(0),axis=0)
    print(datas)
    return datas

def redifine_cluster(data_set,cluster):
    distance = np.zeros((4274,9))
    for i in range(9):
        for j in range(len(data_set[:,1])):
            distance[j][i] = math.sqrt(((data_set[j][0] - cluster[i][0]) ** 2) + ((data_set[j][1] - cluster[i][1]) ** 2) + ((data_set[j][2] - cluster[i][2]) ** 2) + ((data_set[j][3] - cluster[i][3]) ** 2))         

    # print(distance)
    new_cluster = np.argmin(distance,axis = 1)
    
    for i in range(len(new_cluster)):
        data_set[i][12] = new_cluster[i]
    # print(data_set)
    # return data_set    
    
def k_mean_cluster(data_set):
    cluster = []
    for i in range(9):
        mean = [0,0,0,0,0,0,0,0,0,0,0,0]
        length = 0
        for j in range(len(data_set[:,1])):
            if(data_set[j][12] == i):
                length = length + 1
                mean[0] = mean[0] + data_set[j][0]
                mean[1] = mean[1] + data_set[j][1]
                mean[2] = mean[2] + data_set[j][2]
                mean[3] = mean[3] + data_set[j][3]
                mean[4] = mean[4] + data_set[j][4]
                mean[5] = mean[5] + data_set[j][5]
                mean[6] = mean[6] + data_set[j][6]
                mean[7] = mean[7] + data_set[j][7]
                mean[8] = mean[8] + data_set[j][8]
                mean[9] = mean[9] + data_set[j][9]
                mean[10] = mean[10] + data_set[j][10]
                mean[11] = mean[11] + data_set[j][11]
        if(length != 0):
            mean[:] = [x / length for x in mean]

            
        cluster.append(mean)
    # print(cluscluster_mean = np.ones((3,4))ter)
    return cluster


def wcc(dataset,cluster):
    # print(dataset)
    
    distance = np.zeros((len(data_set[:,1]),1))
    for i in range(9):
        for j in range(len(data_set[:,1])):

            if(data_set[j][12] == i):

                distance[j] = ((data_set[j][0] - cluster[i][0]) ** 2) + ((data_set[j][1] - cluster[i][1]) ** 2) + ((data_set[j][2] - cluster[i][2]) ** 2) + ((data_set[j][3] - cluster[i][3]) ** 2) + ((data_set[j][4] - cluster[i][4]) ** 2) + ((data_set[j][5] - cluster[i][5]) ** 2) + ((data_set[j][6] - cluster[i][6]) ** 2) + ((data_set[j][7] - cluster[i][7]) ** 2) + ((data_set[j][8] - cluster[i][8]) ** 2) + ((data_set[j][9] - cluster[i][9]) ** 2) + ((data_set[j][10] - cluster[i][10]) ** 2) + ((data_set[j][11] - cluster[i][11]) ** 2)

    sum = 0
    for j in range(len(distance)):
        sum = sum + distance[j]
    
    return sum

def get_random_cluster(data_set):
    cluster = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    for p in range(9):
        i = np.random.randint(low=5,high=4000)
        cluster = np.vstack([cluster,data_set[i,0:12]])

    cluster = np.delete(cluster,(0),axis=0)
    
    return cluster

def match_array(old,new_arr):
    result = True
    for i in range(9):
        if(old[i][0] == new_arr[i][0] and old[i][1] == new_arr[i][1] and old[i][2] == new_arr[i][2] and old[i][3] == new_arr[i][3] and old[i][4] == new_arr[i][4] and old[i][5] == new_arr[i][5] and old[i][6] == new_arr[i][6] and old[i][7] == new_arr[i][7] and old[i][8] == new_arr[i][8] and old[i][9] == new_arr[i][9] and old[i][10] == new_arr[i][10] and old[i][11] == new_arr[i][11]):
            result = True
        else:
            result = False

        if(result == False):
            break

    return result

if(__name__ == "__main__"):
    data_set = read_data("Vowel")
    cluster_mean = get_random_cluster(data_set)
    
    # while(True):
    len_i = 0
    wcc_data = []
    while(True):
        redifine_cluster(data_set,cluster_mean)
        new_cluster_mean = k_mean_cluster(data_set)
        # print(cluster_mean)
        len_i = len_i + 1

        val = wcc(data_set,cluster_mean)
        wcc_data.append(val)
        print(val)
        if(match_array(cluster_mean,new_cluster_mean)):
            break
        cluster_mean = new_cluster_mean   

    epoch_arr = [x for x in range(len_i)]
    plt.plot(epoch_arr,wcc_data) 
    plt.xlabel("Epoch")
    plt.ylabel("wcss")
    plt.show()   


