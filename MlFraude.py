# %%
# imports
import datetime
from turtle import mode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# %%
TEST_SIZE = 0.2
RANDOM_SEED = 42
# %%
# read csv
df = pd.read_csv('data/treino.csv', sep='|')

df.info()
# %%
# information about data
df.select_dtypes(include="object").describe()
# %%
# drop collums don't usefull
df_train = df.drop(['ssn', 'cc_num', 'first', 'last', 'street', 'zip', 'lat', 'long', 'city_pop', 'acct_num', 'trans_num', 'merchant', 'merch_lat', 'merch_long', 'is_fraud'], axis=1)
df_train.info()
# %%
# function to separate data datetime
def cleaning_data_datetime(data):
    # transform to datetime
    time_transaction = pd.to_datetime(data['trans_time'],format='%H:%M:%S')
    date_transaction = pd.to_datetime(data['trans_date'], format='%Y-%m-%d')

    # apply collumns
    data['hour'] = time_transaction.dt.hour
    data['minute'] = time_transaction.dt.minute
    data['seconds'] = time_transaction.dt.second
    data['year'] = date_transaction.dt.year
    data['month'] = date_transaction.dt.month
    data['day'] = date_transaction.dt.day
    data.drop(['trans_time', 'trans_date'], axis=1, inplace=True)

# %%
cleaning_data_datetime(df_train)
# %%
# function to get age of user
def get_user_age(data):
    year_user = pd.to_datetime(data['dob'], format="%Y-%m-%d")
    data['age'] = (data['year'] - year_user.dt.year)
    data.drop(['dob'], axis=1, inplace=True)
# %%
get_user_age(df_train)
# %%
df_train.info()

# %%
# transform category columns
from sklearn.preprocessing import OneHotEncoder

categorical_data = ['gender', 'city', 'state', 'job', 'profile', 'category']
encoder = OneHotEncoder()

one_hot = encoder.fit_transform(df_train[categorical_data])

encoded_df = pd.DataFrame(one_hot.toarray(), columns=encoder.get_feature_names_out(categorical_data))

df_encoded = pd.concat([df_train.drop(columns=categorical_data), encoded_df], axis=1)

# %%
# split data in train and test
from sklearn.model_selection import train_test_split
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

y_train.fillna(y_train.mean(), inplace=True)
y_test.fillna(y_train.mean(), inplace=True)
# %%
# cleaning null values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
X_train = imputer.fit_transform(X_train)
# %%
# training model
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
y_train_transformed = lab.fit_transform(y_train)
model = RandomForestClassifier()
model.fit(X_train, y_train_transformed)
# %%
# predict values
X_test_transform = imputer.transform(X_test)
y_test = lab.transform(y_test)
pred = model.predict(X_test_transform)
# %%
accuracy_score(y_test, pred)

# %%
# respost 

df = pd.read_csv('data/teste.csv', sep='|')
df_train = df.drop(['ssn', 'cc_num', 'first', 'last', 'street', 'zip', 'lat', 'long', 'city_pop', 'acct_num', 'trans_num', 'merchant', 'merch_lat', 'merch_long'], axis=1)
df_train.info()
cleaning_data_datetime(df_train)
get_user_age(df_train)

one_hot = encoder.transform(df_train[categorical_data])
encoded_df = pd.DataFrame(one_hot.toarray(), columns=encoder.get_feature_names_out(categorical_data))
df_encoded = pd.concat([df_train.drop(columns=categorical_data), encoded_df], axis=1)

X_train = imputer.transform(df_encoded)
#%%
y_sub = model.predict(X_train)
#%%
resposta = pd.DataFrame()
resposta['trans_num'] =  df['trans_num']
resposta['is_fraud'] = y_sub
resposta['is_fraud'].value_counts()
# %%
resposta.to_csv('data/submissao.csv', index=False)
# %%
