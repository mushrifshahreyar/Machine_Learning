from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from keras.models import Sequential

def read_data():
    data_set = pd.read_csv('Google_Stock_Price_Train.csv',index_col="Date",parse_dates=True)
    return data_set

if(__name__ == "__main__") :
    data_set = read_data()
    
    min_max_scaler = MinMaxScaler(feature_range=(0,1));
    data_set_scaled = min_max_scaler.fit_transform(data_set.iloc[:,1:2].values)    
    
    x_train = []
    y_train = []

    for i in range(50,len(data_set)):
        x_train.append(data_set_scaled[i-50:i,0])
        y_train.append(data_set_scaled[i,0])

    
x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# print(x_train.shape)