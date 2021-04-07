from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

def read_data():
    data_set = pd.read_csv('Google_Stock_Price_Train.csv',index_col="Date",parse_dates=True)
    return data_set

def get_train_data(data_set,window_size=50):
    x_train = []
    y_train = []
    
    for i in range(window_size,len(data_set)):
        x_train.append(data_set[i-window_size:i,0])
        y_train.append(data_set[i,0])

    
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train,y_train

def get_test_data(data_set, dataset_test, windowsize):

    
    dataset_total = pd.concat((data_set['Open'], dataset_test['Open']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - windowsize:].values
    inputs = inputs.reshape(-1,1)
    inputs = min_max_scaler.transform(inputs)
    X_test = []
    for i in range(windowsize, windowsize+20):
        X_test.append(inputs[i-windowsize:i, 0])
    X_test = np.array(X_test)
    
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_test

def rnn_lstm(x_train,y_train,custimization):
    model = Sequential()

    if(custimization):
        # Less accuracy
        #input
        model.add(LSTM(units = 60, return_sequences = True, activation='relu',input_shape = (x_train.shape[1], 1)))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 60, activation='relu', return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 80, activation='relu', return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 120, activation='relu',))
        model.add(Dropout(0.2))

        # output
        model.add(Dense(units = 1))
    else:

        #input
        model.add(LSTM(units = 50, return_sequences = True, activation='relu',input_shape = (x_train.shape[1], 1)))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50,  return_sequences = True,))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50,  return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50, ))
        model.add(Dropout(0.2))
        #output
        model.add(Dense(units = 1))

    print(model.summary())

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.fit(x_train, y_train, epochs = 100, batch_size = 32)

    return model


if(__name__ == "__main__") :
    data_set = read_data()
    
    min_max_scaler = MinMaxScaler(feature_range=(0,1));
    data_set_scaled = min_max_scaler.fit_transform(data_set.iloc[:,1:2].values)    
    
    x_train,y_train = get_train_data(data_set_scaled,50)
    
    model = rnn_lstm(x_train,y_train,False)

    #Testing data
    dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')

    real_stock_price = dataset_test.iloc[:, 1:2].values

    X_test = get_test_data(data_set,dataset_test,50)
    
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = min_max_scaler.inverse_transform(predicted_stock_price)

    #Plotting graph
    plt.plot(real_stock_price, color = 'yellow', label = 'Original')
    plt.plot(predicted_stock_price, color = 'violet', label = 'Predicted')
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Google Stock Price')
    plt.legend()
    # plt.show()
    model.save("./GoogleStockPriceModel")
    plt.savefig("Google Stock Price Prediction")