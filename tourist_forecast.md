```python
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LSTM, Bidirectional, RepeatVector, TimeDistributed, BatchNormalization, ReLU, Dropout
from keras.optimizers import RMSprop

from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestClassifier

from pickle import load
```

# 데이터 전처리


```python
df2018 = pd.read_csv('/Users/jaewoo/Documents/4-2/산학프로젝트/지역별 방문자 수(2018).csv')
df2019 = pd.read_csv('/Users/jaewoo/Documents/4-2/산학프로젝트/지역별 방문자 수(2019).csv')
df2020 = pd.read_csv('/Users/jaewoo/Documents/4-2/산학프로젝트/지역별 방문자 수(2020).csv')
df2021 = pd.read_csv('/Users/jaewoo/Documents/4-2/산학프로젝트/지역별 방문자 수(2021).csv')
df2022 = pd.read_csv('/Users/jaewoo/Documents/4-2/산학프로젝트/지역별 방문자 수(2022).csv')

df_origin = pd.concat([df2018, df2019, df2020, df2021, df2022], ignore_index=True)
df_origin = df_origin.astype({'touNum':'int'})
df_origin = df_origin.drop_duplicates(['signguCode', 'touDivCd', 'baseYmd']) # 중복 제거
df_origin.tail()

df = df_origin
df.index =  df['signguCode'].astype(str) + '_' + df['baseYmd']

df_a = df[df['touDivCd'] == 1]
df_a = df_a.rename(columns = {'touNum' : 'touNum_a'})

df_b = df[df['touDivCd'] == 2]
df_b = df_b['touNum']
df_b = df_b.to_frame()
df_b = df_b.rename(columns = {'touNum' : 'touNum_b'})

df_c = df[df['touDivCd'] == 3]
df_c = df_c['touNum']
df_c = df_c.to_frame()
df_c = df_c.rename(columns = {'touNum' : 'touNum_c'})

df_bc = pd.merge(df_b, df_c, left_index=True, right_index=True, how='outer')
df = pd.merge(df_a, df_bc, left_index=True, right_index=True, how='outer')

df['date'] = pd.to_datetime(df['baseYmd'])
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df = df.sort_values(by=['date', 'signguCode'])
df.index = df['date']
df = df.dropna(axis = 0) # 결측치 제거

df['isCovid'] = np.where(('2020-3-22' <= df['date']) & (df['date'] < '2022-4-8'), 1, 0)

df['isHoliday'] = np.where(6 <= df['daywkDivCd'], 1, 0)
holiday = ['2018-01-01', '2018-02-15', '2018-02-16', '2018-02-17', '2018-03-01', '2018-05-05', '2018-05-22', '2018-06-06', '2018-08-15', '2018-09-23', '2018-09-24', '2018-09-25', '2018-10-03', '2018-10-09', '2018-12-25',
          '2019-01-01', '2019-02-04', '2019-02-05', '2019-02-06', '2019-03-01', '2019-05-05', '2019-05-12', '2019-06-06', '2019-08-15', '2019-9-12', '2019-9-13', '2019-9-14', '2019-10-03', '2019-10-09', '2019-12-25',
          '2020-01-01', '2020-01-24', '2020-01-25', '2020-01-26', '2020-03-01', '2020-04-30', '2020-05-05', '2020-06-06', '2020-08-15', '2020-08-17', '2020-09-30', '2020-10-01', '2020-10-02', '2020-10-03', '2020-10-09', '2020-12-25', 
          '2021-01-01', '2021-02-11', '2021-02-12', '2021-02-13', '2021-03-01', '2021-05-05', '2021-05-19', '2021-06-06', '2021-08-15', '2021-09-20', '2021-09-21', '2021-09-22', '2021-10-03', '2021-10-09', '2021-12-25', 
          '2022-01-01', '2022-01-31', '2022-02-01', '2022-02-02', '2022-03-01', '2022-03-09', '2022-04-15', '2022-04-18', '2022-05-01', '2022-05-05', '2022-05-08', '2022-06-01', '2022-06-06', '2022-08-15', '2022-09-09', '2022-09-10', '2022-09-11', '2022-09-12']
for i in range(len(holiday)):
    df.loc[holiday[i],'isHoliday'] = 1
df = df[['month', 'day', 'signguCode', 'daywkDivCd', 'isHoliday', 'isCovid', 'touNum_a', 'touNum_b', 'touNum_c']]
df = df.dropna(axis = 0) # 결측치 제거
df = df.astype({'month':'int', 'day':'int', 'signguCode':'int', 'daywkDivCd':'int', 'isHoliday':'int', 'isCovid':'int', 'touNum_a':'int', 'touNum_b':'int', 'touNum_c':'int'})
# df.to_csv("지역별 방문자 수(2018~2022).csv")
```


```python
df = pd.read_csv('/Users/jaewoo/Documents/4-2/산학프로젝트/지역별 방문자 수(2018~2022).csv')
df['date'] = pd.to_datetime(df['date'])
df.index = df['date']
df = df[['month', 'day', 'signguCode', 'daywkDivCd', 'isCovid', 'isHoliday','touNum_a', 'touNum_b', 'touNum_c']]
df.head(10)
```


```python
temp = df[df['signguCode'] == 11110]
temp = temp[['touNum_a', 'touNum_b', 'touNum_c']]
temp_a = temp['touNum_a']
temp_b = temp['touNum_b']
temp_c = temp['touNum_c']
temp = temp.rename(columns = {'touNum_a':'현지인', 'touNum_b':'외지인', 'touNum_c':'외국인'})

plt.figure(figsize=(15,5))
plt.title("일간 방문자 수")
plt.plot(temp)
plt.legend(temp)
plt.show()

plt.figure(figsize=(15, 5))
plt.title("일간 방문자 수(현지인)")
plt.plot(temp_a)
plt.show()

plt.figure(figsize=(15, 5))
plt.title("일간 방문자 수(외지인)")
plt.plot(temp_b)
plt.show()

plt.figure(figsize=(15, 5))
plt.title("일간 방문자 수(외국인)")
plt.plot(temp_c)
plt.show()
```


```python
from pickle import dump

feature = ['month', 'day', 'signguCode', 'daywkDivCd', 'isHoliday', 'isCovid']
target = ['touNum_b']

def preprocessing(signguCode):
    df_indexed = df[df['signguCode'] == signguCode]
    
    scaler = MinMaxScaler()
    scaler2 = MinMaxScaler()
    
    df_train = df_indexed[df_indexed.index <= '2020-12-31']
    df_val = df_indexed[('2021-01-01' <= df_indexed.index) & (df_indexed.index <= '2021-12-31')]
    df_test = df_indexed['2022-01-01' <= df_indexed.index]
    
    # numpy 배열로 변환
    x_train = df_train[feature].to_numpy()
    y_train = df_train[target].to_numpy()
    
    x_val = df_val[feature].to_numpy()
    y_val = df_val[target].to_numpy()
    
    x_test = df_test[feature].to_numpy()
    y_test = df_test[target].to_numpy()
    
    # 데이터 정규화
    x_train_scaled = scaler.fit_transform(x_train)
    y_train_scaled = scaler2.fit_transform(y_train)
    
    x_val_scaled = scaler.fit_transform(x_val)
    y_val_scaled = scaler2.fit_transform(y_val)
    
    x_test_scaled = scaler.fit_transform(x_test)
    y_test_scaled = scaler2.fit_transform(y_test)
    
    # LSTM 모델에 입력하기 위해 데이터셋 재구성(samples, time steps, features)
    x_train = np.expand_dims(x_train_scaled, axis=0)
    y_train = np.expand_dims(y_train_scaled, axis=0)
    
    x_val = np.expand_dims(x_val_scaled, axis=0)
    y_val = np.expand_dims(y_val_scaled, axis=0)

    x_test = np.expand_dims(x_test_scaled, axis=0)
    y_test = np.expand_dims(y_test_scaled, axis=0)
    
    return x_train, y_train, x_val, y_val, x_test, y_test
```

# BiLSTM 모델 생성


```python
input_dim = len(feature) # 입력 신호의 개수(6)
output_dim = len(target) # 출력 신호의 개수(1)

def BiLSTM():
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(None, input_dim))) # activation func : tanh
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(units=64, activation = ReLU())))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(units=output_dim)))
    
    return model

model = BiLSTM()
model.summary()
```

# 모델 학습


```python
from keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath='checkpoint.h5', monitor='val_loss', save_best_only=True)]

def training_BiLSTM(start, end):
    model = load_model('model_Temp.h5')
    
    for i in range(start, end):
        signguCode = df['signguCode'][i]
        
        x_train, y_train, x_val, y_val, _, _ = preprocessing(signguCode)
        
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1000, batch_size=32, callbacks = callbacks)
        model.save('model_BiLSTM_' + str(signguCode) + '.h5')
        model.save('model_Temp.h5')

signguList = list(df['signguCode'])
# start = 0
start = signguList.index(47830) + 1
end = len(df[df.index == '2018-01-01']) # 학습할 지역의 개수(247)

training_BiLSTM(start, end)
```


```python
def model_BiLSTM(signguCode):
    model = load_model('model_BiLSTM/model_BiLSTM_' + str(signguCode) + '.h5')
    scaler = load(open('scaler/scaler_' + str(signguCode) + '.pkl', 'rb'))
    scaler2 = load(open('scaler/scaler2_' + str(signguCode) + '.pkl', 'rb'))
    
    _, _, _, _, x_test, y_test = preprocessing(signguCode)
    
    y_pred = model.predict(x_test)
    y_pred = scaler2.inverse_transform(y_pred[0])
    
    y_true = y_test
    y_true = scaler2.inverse_transform(y_true[0])
    
    RMSE_list = []
    MAPE_list = []
    
    for signal in range(output_dim):
        true = y_true[:, signal]
        pred = y_pred[:, signal]
        
        plt.figure(figsize=(15,5))
        plt.title("Forecast of BiLSTM Model")
        plt.plot(true, label='true')
        plt.plot(pred, label='pred')
        plt.xlabel("time")
        plt.ylabel("이동 인구")
        plt.legend()
        plt.show()
        
        MSE = mean_squared_error(true, pred)
        RMSE = np.sqrt(MSE)
        RMSE_list.append(RMSE)
        MAPE = np.mean(np.abs((true - pred) / true)) * 100 
        MAPE_list.append(MAPE)
  
    return RMSE_list, MAPE_list

RMSE, MAPE = model_BiLSTM(11140)
print("RMSE :", RMSE)
print("MAPE :", MAPE)
```

# 2022, 2023년의 feature datasets 생성


```python
temp = df[['month', 'day', 'signguCode', 'daywkDivCd', 'isCovid']]
temp = temp[temp.index <= '2018-12-31']

def change_daywkDivcd_2022(x):
    if x == 1:
        x = 6
    elif x == 2:
        x = 7
    else:
        x -= 2
    return x

df_test_2022 = temp[['month', 'day', 'signguCode', 'daywkDivCd', 'isCovid']]
df_test_2022['daywkDivCd'] = df_test_2022['daywkDivCd'].apply(change_daywkDivcd_2022)

df_test_2022['isHoliday'] = np.where(6 <= df_test_2022['daywkDivCd'], 1, 0)
holiday = ['2022-01-01', '2022-01-31', '2022-02-01', '2022-02-02', '2022-03-01', '2022-03-09', '2022-04-15', '2022-04-18', '2022-05-01', '2022-05-05', '2022-05-08', '2022-06-01', '2022-06-06', '2022-08-15', '2022-09-09', '2022-09-10', '2022-09-11', '2022-09-12', '2022-10-03', '2022-10-09', '2022-10-10', '2022-12-25']
for i in range(len(holiday)):
    df_test_2022.loc[holiday[i],'isHoliday'] = 1
df_test_2022 = df_test_2022.dropna(axis = 0) # 결측치 제거
df_test_2022 = df_test_2022.astype({'month':'int', 'day':'int', 'signguCode':'int', 'daywkDivCd':'int', 'isHoliday':'int', 'isCovid':'int'})
df_test_2022.to_csv("df_test_2022.csv", index = False)

def change_daywkDivcd_2023(x):
    if x == 1:
        x = 7
    else:
        x -= 1
    return x

df_test_2023 = temp[['month', 'day', 'signguCode', 'daywkDivCd', 'isCovid']]
df_test_2023['daywkDivCd'] = df_test_2023['daywkDivCd'].apply(change_daywkDivcd_2023)

df_test_2023['isHoliday'] = np.where(6 <= df_test_2023['daywkDivCd'], 1, 0)
holiday = ['2023-01-01', '2023-01-21', '2023-01-22', '2023-01-23',  '2023-01-24', '2023-03-01', '2023-05-05', '2023-03-01', '2023-05-05', '2023-05-27', '2023-06-06', '2023-08-15', '2023-09-28', '2023-09-29', '2023-09-30', '2023-10-03', '2023-10-09', '2023-12-25']
for i in range(len(holiday)):
    df_test_2023.loc[holiday[i],'isHoliday'] = 1
df_test_2023 = df_test_2023.dropna(axis = 0) # 결측치 제거
df_test_2023 = df_test_2023.astype({'month':'int', 'day':'int', 'signguCode':'int', 'daywkDivCd':'int', 'isHoliday':'int', 'isCovid':'int'})
df_test_2023.to_csv("df_test_2023.csv", index =False)
```


```python
def get_df_result(df, signguCode):
    model = load_model('model_BiLSTM/model_BiLSTM_' + str(signguCode) + '.h5')
    
    scaler = load(open('scaler/scaler_' + str(signguCode) + '.pkl', 'rb'))
    scaler2 = load(open('scaler/scaler2_' + str(signguCode) + '.pkl', 'rb'))
    
    df_test = df[df['signguCode'] == signguCode]
    df_test = df_test.reset_index()
    
    x_test = df_test[feature].to_numpy()
    x_test_scaled = scaler.fit_transform(x_test)
    x_test = np.expand_dims(x_test_scaled, axis=0)
    
    y_pred = model.predict(x_test)
    y_pred = scaler2.inverse_transform(y_pred[0])
    
    df_pred = pd.DataFrame(y_pred, columns = ['touNum_b'])
    
    df_temp = df_test[['signguCode', 'month', 'day']]
    df_result = pd.concat([df_temp, df_pred], axis=1)
    
    return df_result
```


```python
df_2022 = pd.read_csv('/Users/jaewoo/Documents/4-2/산학프로젝트/df_test_2022.csv')
df_2023 = pd.read_csv('/Users/jaewoo/Documents/4-2/산학프로젝트/df_test_2023.csv')

df_result_2022 = pd.read_csv('/Users/jaewoo/Documents/4-2/산학프로젝트/df_result_2022_1.csv')
df_result_2023 = pd.read_csv('/Users/jaewoo/Documents/4-2/산학프로젝트/df_result_2023_1.csv')

signguList = list(df['signguCode'])
# start = 0
start = signguList.index(43730) + 1
end = 247 # 지역의 개수

for i in range(start, end):
    df_temp = get_df_result(df_2022, signguList[i])
    df_result_2022 = pd.concat([df_result_2022, df_temp], axis=0)
    df_result_2022.to_csv("df_result_2022.csv", index = False)
    
    df_temp = get_df_result(df_2023, signguList[i])
    df_result_2023 = pd.concat([df_result_2023, df_temp], axis=0)
    df_result_2023.to_csv("df_result_2023.csv", index = False)
    
    print(signguList[i], "(", i + 1, "/", end, ")")
```


```python
df_result_2022 = pd.read_csv('/Users/jaewoo/Documents/4-2/산학프로젝트/df_result_2022.csv')
df_result_2023 = pd.read_csv('/Users/jaewoo/Documents/4-2/산학프로젝트/df_result_2023.csv')

df_result_2022 = df_result_2022.rename(columns = {'signguCode' : 'region_code', 'touNum_b' : 'visitors_num'})
df_result_2023 = df_result_2023.rename(columns = {'signguCode' : 'region_code', 'touNum_b' : 'visitors_num'})

def change(x):
    if x < 10:
        x = '0' + str(x)
    else:
        str(x)
    return x

df_result_2022['month'] = df_result_2022['month'].apply(change).astype(str)
df_result_2022['day'] = df_result_2022['day'].apply(change).astype(str)
df_result_2022['date_str'] = '2022-' + df_result_2022['month'] + '-' + df_result_2022['day']
df_result_2022 = df_result_2022.astype({'visitors_num':'int'})
df_result_2022 = df_result_2022.astype({'date_str':'str'})
df_result_2022 = df_result_2022[['date_str', 'region_code', 'visitors_num']]
df_result_2022.to_csv("result_2022.csv", index = False)

df_result_2023['month'] = df_result_2023['month'].apply(change).astype(str)
df_result_2023['day'] = df_result_2023['day'].apply(change).astype(str)
df_result_2023['date_str'] = '2023-' + df_result_2023['month'] + '-' + df_result_2023['day']
df_result_2023 = df_result_2023.astype({'visitors_num':'int'})
df_result_2023 = df_result_2023.astype({'date_str':'str'})
df_result_2023 = df_result_2023[['date_str', 'region_code', 'visitors_num']]
df_result_2023.to_csv("result_2023.csv", index = False)
df_result_2022.head()
```

# 예측 결과


```python
def get_result(month, day, signguCode, daywkDivCd, isHoliday, isCovid=0):
    model = load_model('model_BiLSTM/model_BiLSTM_' + str(signguCode) + '.h5')
    scaler = load(open('scaler/scaler_' + str(signguCode) + '.pkl', 'rb'))
    scaler2 = load(open('scaler/scaler2_' + str(signguCode) + '.pkl', 'rb'))
    
    x = np.array([[month, day, signguCode, daywkDivCd, isHoliday, isCovid]])
    x_scaled = scaler.fit_transform(x)
    x = np.expand_dims(x_scaled, axis=0)
    y = model.predict(x)

    return int(scaler2.inverse_transform(y[0]))

result = get_result(month=1, day=5, signguCode=11140, daywkDivcd=0, isHoliday=0)
print(result)
```


```python
input_dim = len(feature) # 입력 신호의 개수
output_dim = len(target) # 출력 신호의 개수

def lstm():
    model = Sequential()
    model.add(LSTM(units=200, return_sequences=True, input_shape=(None, input_dim)))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(TimeDistributed(Dense(units=output_dim)))
    
    return model

model = lstm()
model.summary()
```


```python
from keras.callbacks import EarlyStopping, ModelCheckpoint
callbacks = [EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath='model_LSTM_best.h5', monitor='val_loss', save_best_only=True)]

model = lstm()
signguCode = df['signguCode'][0]
x_train, y_train, x_val, y_val, _, _ = preprocessing(signguCode)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=32, callbacks = callbacks)
model.save('model_LSTM_11110.h5')
```


```python
def model_LSTM(signguCode):
    model = load_model('model_LSTM_11110.h5')
    scaler = load(open('scaler/scaler_' + str(signguCode) + '.pkl', 'rb'))
    scaler2 = load(open('scaler/scaler2_' + str(signguCode) + '.pkl', 'rb'))
    
    _, _, _, _, x_test, y_test = preprocessing(signguCode)
    
    y_pred = model.predict(x_test)
    y_pred = scaler2.inverse_transform(y_pred[0])
    
    y_true = y_test
    y_true = scaler2.inverse_transform(y_true[0])
    
    RMSE_list = []
    MAPE_list = []
    
    for signal in range(output_dim):
        true = y_true[:, signal]
        pred = y_pred[:, signal]
        
        plt.figure(figsize=(15,5))
        plt.title("Forecast of BiLSTM Model")
        plt.plot(true, label='true')
        plt.plot(pred, label='pred')
        plt.xlabel("time")
        plt.ylabel("이동 인구")
        plt.legend()
        plt.show()
        
        MSE = mean_squared_error(true, pred)
        RMSE = np.sqrt(MSE)
        RMSE_list.append(RMSE)
        MAPE = np.mean(np.abs((true - pred) / true)) * 100 
        MAPE_list.append(MAPE)
  
    return RMSE_list, MAPE_list

RMSE_LSTM, MAPE_LSTM = model_LSTM(11110)
print("RMSE_LSTM :", RMSE_LSTM)
print("MAPE_LSTM :", MAPE_LSTM)
```


```python
import itertools

def get_pdq(df):
    p = q = range(0, 7)
    d = range(0, 3)
    pdq = []
    params = list(itertools.product(p,d,q))
    min_aic = 1e9
    
    for param in params:
        model = ARIMA(df, order=param)
        model_fit = model.fit()
        if (model_fit.aic < min_aic):
            min_aic = model_fit.aic
            pdq = param
    
    return pdq

def model_ARIMA(signguCode):
    df_indexed = df[df['signguCode'] == signguCode]
    
    RMSE_list = []
    MAPE_list = []
    
    for i in range(1):
        df_train = df_indexed[df_indexed.index <= '2021-12-31']
        df_test = df_indexed['2022-01-01' <= df_indexed.index]
        pdq = get_pdq(df_train)
        
        model = ARIMA(df_train, order=pdq, enforce_stationarity=False)
        model_fit = model.fit()
        pred = model_fit.forecast(steps=len(df_test)).to_numpy()
        true = df_test.to_numpy()
        
        plt.figure(figsize=(15,5))
        plt.title("Forecast of ARIMA Model")
        plt.plot(true, label='true')
        plt.plot(pred, label='pred')
        plt.xlabel("time")
        plt.ylabel("이동 인구")
        plt.legend()
        plt.show()
        
        MSE = mean_squared_error(true, pred)
        RMSE = np.sqrt(MSE)
        RMSE_list.append(RMSE)
        MAPE = np.mean(np.abs((true - pred) / true)) * 100 
        MAPE_list.append(MAPE)
        
    return RMSE_list, MAPE_list
```


```python
import itertools

def get_optimal(df):
    p = d = q = range(0, 7)
    params = list(itertools.product(p,d,q))
    seasonal_params = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p,d,q))]
    min_aic = 1e9
    
    pdq = []
    seasonal_pdq = []
    
    for param in params:
        for seasonal_param in seasonal_params:
            model = SARIMAX(df, order=param, seasonal_order=seasonal_param)
            model_fit = model.fit()
            if (model_fit.aic < min_aic):
                min_aic = model_fit.aic
                pdq = param
                seasonal_pdq = seasonal_param
    
    return pdq, seasonal_pdq

def model_SARIMAX(signguCode):
    df_indexed = df[df['signguCode'] == signguCode]
    
    df_indexed_list = []
    df_indexed_list.append(df_indexed[['touNum_a']])
    df_indexed_list.append(df_indexed[['touNum_b']])
    df_indexed_list.append(df_indexed[['touNum_c']])
    
    RMSE_list = []
    MAPE_list = []
    
    for i in range(1):
        df_train = df_indexed_list[i][df_indexed_list[i].index <= '2021-12-31']
        df_test = df_indexed_list[i]['2022-01-01' <= df_indexed_list[i].index]
        pdq, seasonal_pdq = get_optimal(df_train)
        
        model = SARIMAX(df_train, order = pdq, seasonal_order = seasonal_pdq, enforce_stationarity=False)
        model.initialize_approximate_diffuse()
        model_fit = model.fit()
        pred = model_fit.forecast(steps=len(df_test)).to_numpy()
        true = df_test.to_numpy()
        
        plt.figure(figsize=(15,5))
        plt.title("Forecast of SARIMA Model")
        plt.plot(true, label='true')
        plt.plot(pred, label='pred')
        plt.xlabel("time")
        plt.ylabel("이동 인구")
        plt.legend()
        plt.show()
        
        MSE = mean_squared_error(true, pred)
        RMSE = np.sqrt(MSE)
        RMSE_list.append(RMSE)
        MAPE = np.mean(np.abs((true - pred) / true)) * 100 
        MAPE_list.append(MAPE)
        
    return RMSE_list, MAPE_list
```


```python
def preprocessing_RF(signguCode):
    df_indexed = df[df['signguCode'] == signguCode]
    df_train = df_indexed[df_indexed.index <= '2021-12-31']
    df_test = df_indexed['2022-01-01' <= df_indexed.index]

    x_train = df_train[['month', 'day', 'signguCode', 'daywkDivCd', 'isHoliday', 'isCovid']].to_numpy()
    y_train = df_train[['touNum_b']].to_numpy()

    x_test = df_test[['month', 'day', 'signguCode', 'daywkDivCd', 'isHoliday', 'isCovid']].to_numpy()
    y_test = df_test[['touNum_b']].to_numpy()
    
    return x_train, y_train, x_test, y_test

def model_RandomForest(signguCode):
    x_train, y_train, x_test, y_test = preprocessing_RF(signguCode)
    
    model = RandomForestClassifier(n_estimators = 1000)
    model.fit(x_train, y_train)
    
    pred = model.predict(x_test)
    true = y_test
    
    RMSE_list = []
    MAPE_list = []
    
    MSE = mean_squared_error(true, pred)
    RMSE = np.sqrt(MSE)
    RMSE_list.append(RMSE)
    MAPE = np.mean(np.abs((true - pred) / true)) * 100 
    MAPE_list.append(MAPE)
    
    return RMSE_list, MAPE_list
```


```python
warnings.filterwarnings('ignore')

RMSE_ARIMA, MAPE_ARIMA = model_ARIMA(11110)
print("RMSE_ARIMA :", RMSE_ARIMA)
print("MAPE_ARIMA :", MAPE_ARIMA)
```


```python
RMSE_RandomForest, MAPE_RandomForest = model_RandomForest(11110)
print("RMSE_RandomForest :", RMSE_RandomForest)
print("MAPE_RandomForest :", MAPE_RandomForest)
```
