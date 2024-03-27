# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



def treat_train_data(df):
  df.dropna(subset=['is_fraud'],inplace=True)
  X = df.drop(['is_fraud', 'ssn', 'first', 'last','cc_num', 'street', 'acct_num', 'profile', 'trans_num'], axis=1)
  # Substituindo os valores nulos por 0
  X['zip'] = X['zip'].fillna(0)
  X['lat'] = X['lat'].fillna(0)
  X['long'] = X['long'].fillna(0)
  X['merch_lat'] = X['merch_lat'].fillna(0)
  X['merch_long'] = X['merch_long'].fillna(0)

  X['hora'] =pd.to_datetime(X['trans_time'],format='%H:%M:%S').dt.hour
  X['minuto'] =pd.to_datetime(X['trans_time'],format='%H:%M:%S').dt.minute
  X['segundo'] =pd.to_datetime(X['trans_time'],format='%H:%M:%S').dt.second
  X['ano'] =pd.to_datetime(X['trans_date'], format='%Y-%m-%d').dt.year
  X['mês'] =pd.to_datetime(X['trans_date'], format='%Y-%m-%d').dt.month
  X['dia'] =pd.to_datetime(X['trans_date'], format='%Y-%m-%d').dt.day

  X = X.drop(['trans_time','trans_date'],axis=1)

  # Tratando os dados categoricos
  categorys_columns = ['gender', 'city', 'state', 'job', 'dob', 'category', 'merchant']
  X_encoded = encoder.fit_transform(X[categorys_columns])
  y = df['is_fraud']
  return (X_encoded, y)

def treat_test_data(df_sub):
  X = df_sub.drop(['first','last','ssn','cc_num','profile','trans_num','street','acct_num'],axis=1)
  X['zip'] = X['zip'].fillna(0)
  X['lat'] = X['lat'].fillna(0)
  X['long'] = X['long'].fillna(0)
  X['merch_lat'] = X['lat'].fillna(0)
  X['merch_long'] = X['long'].fillna(0)
  X['hora'] =pd.to_datetime(X['trans_time'],format='%H:%M:%S').dt.hour
  X['minuto'] =pd.to_datetime(X['trans_time'],format='%H:%M:%S').dt.minute
  X['segundo'] =pd.to_datetime(X['trans_time'],format='%H:%M:%S').dt.second
  X['ano'] =pd.to_datetime(X['trans_date'], format='%Y-%m-%d').dt.year
  X['mês'] =pd.to_datetime(X['trans_date'], format='%Y-%m-%d').dt.month
  X['dia'] =pd.to_datetime(X['trans_date'], format='%Y-%m-%d').dt.day
  X = X.drop(['trans_time','trans_date'],axis=1)
  categorys_columns = ['gender', 'city', 'state', 'job', 'dob', 'category', 'merchant']
  X_encoded = encoder.transform(X[categorys_columns])
  return X_encoded

def response(df, model): 
  X = treat_test_data(df)
  y = model.predict(X)
  response = pd.DataFrame()
  response['trans_num'] = df['trans_num']
  response['is_fraud'] = y
  response.to_csv('submissão.csv', index=False)

df_train = pd.read_csv('data/treino.csv', sep='|')
df_test = pd.read_csv('data/teste.csv', sep='|')

# Removendo colunas inutilizadas
X_encoded, y = treat_train_data(df_train, False)
# Separando os dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
# Escalando os Dados
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Treinando o modelo
model = RandomForestClassifier(random_state=42)

model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
accuracy = model.score(X_test_scaled, y_test)
print(accuracy)

from sklearn.model_selection import GridSearchCV

forest = RandomForestClassifier()

param_grid = {
  "n_estimators": [3,10,30],
  "max_feature": [2,4,6,8]
}

grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)

grid_search.fit(X_train_scaled, y_train)
#response(df_test, model=model)