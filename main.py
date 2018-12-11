import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pandasql import sqldf
q = lambda q: sqldf(q, globals())
from pandas import concat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import linear_model, neighbors, tree, svm, ensemble
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
train = read_csv("house_prices/train.csv")._get_numeric_data()

X_df = train.drop(["SalePrice"], axis=1)
X_df = X_df.fillna(train.mean())
scaler = StandardScaler()
train_X_temp = scaler.fit_transform(X_df)
train_y = train['SalePrice'].values

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
np.random.seed(3)
model = Sequential()
first_layer_size = 20
first_hidden_layer_size = 10
last_hidden_layer_size = 10
hidden_layers = 2
model.add(Dense(first_layer_size, input_dim=37, activation='relu'))
model.add(Dense(first_hidden_layer_size, activation='relu'))
model.add(Dense(last_hidden_layer_size, activation='relu'))
model.add(Dense(1))
early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_X_temp, train_y, epochs=150, batch_size=10, verbose=0, callbacks=[early_stop])
shorter_models = []
for i in range(0, last_hidden_layer_size):
    new_model = Sequential()
    new_model.add(Dense(first_layer_size, input_dim=37, activation='relu'))
    new_model.add(Dense(first_hidden_layer_size))
    new_model.add(Dense(1))
    # Compile model
    new_model.compile(loss='mean_squared_error', optimizer='adam')
    for j in range(0, hidden_layers):
        new_model.layers[j].set_weights(model.get_weights()[(2*j):(2*(j+1))])
    new_model.layers[hidden_layers].set_weights([np.vstack(model.get_weights()[2*j][:,i]), np.array([model.get_weights()[(2*j)+1][i]])])
    shorter_models.append(new_model)
df = pd.DataFrame()
for i in range(0, last_hidden_layer_size):
    df['NN_feature_{}'.format(i)] = shorter_models[i].predict(train_X_temp).flatten()
train_X = concat([X_df, df], axis=1)
train_X = scaler.fit_transform(train_X)

# model = ensemble.GradientBoostingRegressor(n_estimators=1000)
model = XGBRegressor(max_depth=2, n_estimators=200)
cross_val_score(model, train_X, train_y, cv = KFold(n_splits=5, random_state=22), scoring='neg_mean_squared_error')
