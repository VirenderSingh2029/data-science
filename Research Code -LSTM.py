# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:45:11 2019

@author: Virender SIngh
"""

from __future__ import print_function
import os
import sys
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
import datetime

os.chdir('E:/PROJECT')

df = pd.read_csv('dataset/GuangzhouPM20100101_20151231.csv')

print('Shape of the dataframe:', df.shape)


#Let's see the first five rows of the DataFrame
df.head()


"""
 NaN values in column pm2.5 are dropped.
"""
df.dropna(subset=['pm2.5'], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)


df['datetime'] = df[['year', 'month', 'day', 'hour']].apply(lambda row: datetime.datetime(year=row['year'], month=row['month'], day=row['day'],
                                                                                          hour=row['hour']), axis=1)
df.sort_values('datetime', ascending=True, inplace=True)



#  here a box plot has been drawn  to visualize the central tendency and dispersion of PRES
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(df['pm2.5'])
g.set_title('Box plot of pm2.5')





plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['pm2.5'])
g.set_title('pm2.5 Time Series')
g.set_xlabel('Index')
g.set_ylabel('readings of pm2.5')




# plot the series for six months to check if any pattern exists in the dataset.
plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['pm2.5'].loc[df['datetime']<=datetime.datetime(year=2010,month=6,day=30)], color='r')
g.set_title('pm2.5 during 2010')
g.set_xlabel('Index')
g.set_ylabel('pm2.5 readings')






#Let's zoom in on one month.
plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['pm2.5'].loc[df['datetime']<=datetime.datetime(year=2010,month=1,day=31)], color='g')
g.set_title('pm2.5 during Jan 2010')
g.set_xlabel('Index')
g.set_ylabel('pm2.5 readings')



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_pm2.5'] = scaler.fit_transform(np.array(df['pm2.5']).reshape(-1, 1))


"""
splitting the dataset into train and validation
time period if from
Jan 1st, 2010 to Dec 31st, 2014
first fours years - 2010 to 2013 is used as train and
2014 is for validation. """


split_date = datetime.datetime(year=2014, month=1, day=1, hour=0)
df_train = df.loc[df['datetime']<split_date]
df_val = df.loc[df['datetime']>=split_date]
print('Shape of train:', df_train.shape)
print('Shape of test:', df_val.shape)




#First five rows of train

df_train.head()


#First five rows of validation

df_val.head()

#Resetting the indices of the validation set

df_val.reset_index(drop=True, inplace=True)





"""
The train and validation time series of scaled_pm2.5 is also plotted.
"""

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_train['scaled_pm2.5'], color='b')
g.set_title('Time series  in train set of scaled pm2.5')
g.set_xlabel('Index')
g.set_ylabel('Scaled pm2.5 readings')

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_val['scaled_pm2.5'], color='r')
g.set_title('Time series  in validation set of scaled pm2.5')
g.set_xlabel('Index')
g.set_ylabel('Scaled pm2.5 readings')



 
"""
           ts: original time series
           nb_timesteps: number of time steps in the regressors
    Output: 
           X: 2-D array of regressors
           y: 1-D array of target 
"""

def makeXy(ts, nb_timesteps):
   
    X = []
    y = []
    for i in range(nb_timesteps, ts.shape[0]):
        X.append(list(ts.loc[i-nb_timesteps:i-1]))
        y.append(ts.loc[i])
    X, y = np.array(X), np.array(y)
    return X, y




X_train, y_train = makeXy(df_train['scaled_pm2.5'], 7)
print('Shape of train arrays:', X_train.shape, y_train.shape)





X_val, y_val = makeXy(df_val['scaled_pm2.5'], 7)
print('Shape of validation arrays:', X_val.shape, y_val.shape)





X_train, X_val = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
print('Shape of arrays after reshaping:', X_train.shape, X_val.shape)


# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
...
1
2
3
4
5
# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential

from keras.layers import Dense, Dropout, Input
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint





input_layer = Input(shape=(7,1), dtype='float32')


""" The LSTM layers are defined for seven timesteps. In this example, two LSTM layers
 are stacked. The first LSTM returns the output from each all seven timesteps. 
 This output is a sequence and is fed to the second LSTM which
 returns output only from the last step. 
The first LSTM has sixty four hidden neurons in each timestep.
 Hence the sequence returned by the first LSTM has sixty four features"""
 
 
lstm_layer1 = LSTM(64, input_shape=(7,1), return_sequences=True)(input_layer)
lstm_layer2 = LSTM(32, input_shape=(7,64), return_sequences=False)(lstm_layer1)


dropout_layer = Dropout(0.2)(lstm_layer2)



#Finally the output layer gives prediction for the next day's air pressure.




output_layer = Dense(1, activation='linear')(dropout_layer)
import os

ts_model = Model(inputs=input_layer, outputs=output_layer)
ts_model.compile(loss='mean_absolute_error', optimizer='adam')#SGD(lr=0.001, decay=1e-5))
ts_model.summary()


save_weights_at = os.path.join('C:/Users/hi/Desktop/kerasmodels', 'PM2.5_LSTM_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)


ts_model.fit(x=X_train, y=y_train, batch_size=16, epochs=20,
             verbose=1, callbacks=[save_best], validation_data=(X_val, y_val),
             shuffle=True)






# error solved here 

best_model = load_model(os.path.join('C:/Users/hi/Desktop/kerasmodels/', 'PM2.5_LSTM_weights.19-0.0118.hdf5'))
preds = best_model.predict(X_val)
pred_pm25 = scaler.inverse_transform(preds)
pred_pm25 = np.squeeze(pred_pm25)


  
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(df_val['pm2.5'].loc[7:], pred_pm25)
print('MAE for the validation set:', round(mae, 4))





plt.figure(figsize=(5.5, 5.5))
plt.plot(range(50), df_val['pm2.5'].loc[7:56], linestyle='-', marker='*', color='r')
plt.plot(range(50), pred_pm25[:50], linestyle='-', marker='.', color='b')
plt.legend(['Actual','Predicted'], loc=2)
plt.title('Actual vs Predicted pm2.5')
plt.ylabel('pm2.5')
plt.xlabel('Index')


