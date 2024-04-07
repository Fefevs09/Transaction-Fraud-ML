# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import neighbors
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

TEST_SIZE = 0.2
RANDOM_SEED = 42
# read csv
url = "https://raw.githubusercontent.com/Fefevs09/Transaction-Fraud-ML/main/data/treino.csv"
df = pd.read_csv(url,sep='|', encoding="utf-8" )
print(df.info())

# information about data
df.select_dtypes(include="object").describe()

# drop collums don't usefull
df = df.dropna(subset="is_fraud")
df_train = df.drop(['ssn', 'cc_num', 'first', 'last', 'street', 'zip', 'unix_time', 'lat', 'long', 'acct_num', 'trans_num', 'merchant', 'merch_lat', 'merch_long', 'is_fraud'], axis=1)
df_train.info()

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

cleaning_data_datetime(df_train)
# function to get age of user
def get_user_age(data):
    year_user = pd.to_datetime(data['dob'], format="%Y-%m-%d")
    data['age'] = (data['year'] - year_user.dt.year)
    data.drop(['dob'], axis=1, inplace=True)
get_user_age(df_train)
# df_train.info()

# set street names to num
def set_street_num(data):
    data['street'] = data['street'].str.split().str[0]
    data['street'] = data['street'].astype(int)

# set_street_num(df_train)

# transform category columns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Supondo que 'df_train' seja o seu DataFrame original

# Extrair colunas categóricas
categorical_data = ['gender', 'job', 'profile', 'category', 'city', 'state']

# Inicializar LabelEncoder e OneHotEncoder
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder()

# Aplicar LabelEncoder para cada coluna categórica
for column in categorical_data:
    df_train[column] = label_encoder.fit_transform(df_train[column])

# # Aplicar OneHotEncoder para todas as colunas categóricas
# encoded_features = onehot_encoder.fit_transform(df_train[categorical_data])

# # Converter as features one-hot em DataFrame pandas
# encoded_df = pd.DataFrame(encoded_features.toarray(), columns=onehot_encoder.get_feature_names_out(categorical_data))

# # Concatenar o DataFrame one-hot com o DataFrame original
# df_encoded = pd.concat([df_train.drop(columns=categorical_data), encoded_df], axis=1)

# # Separar os dados em conjuntos de treino e teste
from sklearn.model_selection import train_test_split
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# cleaning null values
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

imputer = SimpleImputer()
X_train_transform = imputer.fit_transform(X_train)

lab = LabelEncoder()
y_train_transformed = lab.fit_transform(y_train)

X_test_transform = imputer.transform(X_test)
y_test_transform = lab.transform(y_test)

# training models
# algorithms machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# model = RandomForestClassifier()
# model.fit(X_train, y_train_transformed)

models = {
    "KNN": KNeighborsClassifier(),
    "Logist Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier()
}

def fit_and_score(models, X_train, X_test, y_train, y_test):
    models_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        models_scores[name] = model.score(X_test, y_test)
    return models_scores


score = fit_and_score(models=models, X_train=X_train_transform, X_test=X_test_transform, y_train=y_train_transformed, y_test=y_test_transform)
print(score)

# hyperparameters adjustements and cross validation
from sklearn.model_selection import RandomizedSearchCV

train_score = []
test_score = []

rf_grid = {
    "n_estimators": np.arange(10, 1000, 50),
    "max_depth": [None, 3,5,10],
    "min_samples_split": np.arange(2,20,2),
    "min_samples_leaf": np.arange(1, 20, 2)
}
rscv_rf = RandomizedSearchCV(RandomForestClassifier(),
                             param_distributions=rf_grid,
                             cv=5,
                             n_iter=20,
                             verbose=True)

rscv_rf.fit(X_train_transform, y_train_transformed)

print(rscv_rf.best_params_)
print(rscv_rf.score(X_test_transform, y_test_transform))

# response
url_test = "https://raw.githubusercontent.com/Fefevs09/Transaction-Fraud-ML/main/data/teste.csv"
df = pd.read_csv(url_test, sep='|', encoding="utf-8")
df

df_test = df.drop(['ssn', 'cc_num', 'first', 'last', 'street', 'zip', 'unix_time', 'lat', 'long', 'acct_num', 'trans_num', 'merchant', 'merch_lat', 'merch_long'], axis=1)
df_test.info()
cleaning_data_datetime(df_test)
get_user_age(df_test)

# set_street_num(df_test)
for column in categorical_data:
    df_test[column] = label_encoder.fit_transform(df_test[column])

#
# y_sub = knn.predict(X_train)
y_sub = rscv_rf.predict(df_test)

# generate response dataframe
response = pd.DataFrame()
response['trans_num'] =  df['trans_num']
response['is_fraud'] = y_sub
# response['is_fraud'].value_counts()
response.to_csv('data/submissao.csv', index=False)