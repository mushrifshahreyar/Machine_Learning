import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import os.path

def read_document():
    print("Reading Document")
    X = np.array([[0,0,0,0,0]])
    with open('airfoil_self_noise.dat') as f:
        for line in f:
            data = line.split()
            temp = np.array([])
            for i in range(0,5):
                temp = np.append(temp,float(data[i]))
            X = np.vstack([X,temp])
        X = np.delete(X,(0),axis=0)
    
    X1 = np.ones((len(X[:,1]),1))
    # X = normalize(X,axis=0)
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
    X = np.concatenate((X1,X),axis=1)
    
    Y = np.array([0])
    with open('airfoil_self_noise.dat') as f:
        for line in f:
            data = line.split()
            Y = np.vstack([Y,np.array([float(data[5])])])
        Y = np.delete(Y,(0),axis=0)
    
    print("Finished Reading")
    return X,Y

def gradient_descent(theta,X,Y,temp,Learner_rate):
    for i in range(6):
        val = 0
        for j in range(len(X[:,1])):
            val = val + (temp[j] * X[j][i])
        
        val = val/len(X[:,1])
        theta[i] = theta[i] - (Learner_rate * val)

    return theta

def linear_reg(theta,X,Y,epoch,Learner_rate):
    
    cost_arr = []
    for i in range(epoch):
        Hx = np.dot(X,theta)    
        temp = np.subtract(Hx,Y)
        
        cost = 0

        for j in range(len(X[:,1])):
            cost = cost + (temp[j][0] ** 2)

        cost = cost / (2 * len(X[:,1]))
        cost_arr.append(cost)

        theta = gradient_descent(theta,X,Y,temp,Learner_rate)
        print(cost)
    
    # print(cost_arr)
    epoch_arr = [x for x in range(epoch)]
    plt.plot(epoch_arr,cost_arr) 
    plt.xlabel("Epoch")
    plt.ylabel("cost")
    # print(theta)
    plt.show()   

            

if __name__ == "__main__":
    X,Y = read_document()
    theta = np.array([[0]])
    File_name = "./theta.txt"
    if(os.path.isfile(File_name)):
        with open("theta.txt","r") as f:
            for line in f:
                data = line.strip()
                print(data)
                data = float(data)
                theta = np.vstack([theta,data])
            theta = np.delete(theta,(0),axis=0)
        print(theta)
        f.close()
    else:
        theta = np.zeros((6,1))

    # print(X)
    linear_reg(theta,X,Y,5000,0.05)
    fd = open("theta.txt","w")
    for x in range(6):
        fd.write(str(theta[x][0]) + "\n")
    print(theta)
    fd.close()