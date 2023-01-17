pip install tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test)=keras.datasets.boston_housing.load_data()
def create_model(hidden_layers,neurons_per_layers):
    for i in range(hidden_layers):
        model=keras.Sequential()
        model.add(layers.Dense(neurons_per_layers,activation='relu',input_shape=x_train.shape[1]))
        model.add(layers.Dense[1])
        model.compile(optimizer='adam',loss='mean_squared_error', metrics=['mae'])
    return model
hidden_layers=[1,2,3]
neurons_per_layers=[32,64,128]
for hidden in hidden_layers:
    for neurons in neurons_per_layers:
        model=create_model(hidden,neurons)
        history=model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test))
import numpy as np
def plot_history(history):
    plt.figure(figsize=[6,7])
    plt.subplot(1,2,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error[1000$]')
    plt.plot(history.epoch,np.array(history.history['mae'],labels='Train loss'))
    plt.plot(history.epoch,np.array(history.history['val_mae'],labels='val loss'))
    plt.legend()
    plt.subplot(1,2,2)
    plt.ylabel('Mean Absolute Error[1000$]')
    plt.plot(history.epoch,np.array(history.history['loss'],labels='Train loss'))
    plt.plot(history.epoch,np.array(history.history['val_loss'],labels='val loss'))
    plt.legend()
    plt.show()
    
plot_history(history)
y_pred.predict(y_test)
print(real=y_test)
print(pred=y_pred)