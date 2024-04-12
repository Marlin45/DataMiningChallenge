import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from tensorflow import keras
import keras_tuner
from keras.models import Model
from keras import layers
from keras import Input
from keras.layers import Dense, LeakyReLU, ReLU, Conv1D


def prep_demo_data():

    X_train = pd.read_csv("./train/train_demos.csv")
    X_test = pd.read_csv("./test/test_demos.csv")

    X_train = X_train.set_index('patient_id')
    X_test = X_test.set_index('patient_id')
    cat = ['gender', 'insurance', 'marital_status', 'ethnicity']
    # ONE HOT
    enc = OneHotEncoder()
    X_train_enc = enc.fit_transform(X_train[cat])
    X_test_enc = enc.transform(X_test[cat])
    X_train_enc = pd.DataFrame.sparse.from_spmatrix(X_train_enc)
    X_test_enc = pd.DataFrame.sparse.from_spmatrix(X_test_enc)
    X_train_enc.index = X_train.index
    X_test_enc.index = X_test.index
    
    X_train = pd.concat([X_train.drop(cat, axis=1), X_train_enc], axis=1)
    X_test = pd.concat([X_test.drop(cat, axis=1), X_test_enc], axis=1)

    X_train['admittime'] = X_train.apply(lambda x: time.mktime(pd.Timestamp(x['admittime']).timetuple()), axis=1)
    X_train['admittime'] = X_train['admittime'] - X_train['admittime'].min()

    X_test['admittime'] = X_test.apply(lambda x: time.mktime(pd.Timestamp(x['admittime']).timetuple()), axis=1)
    X_test['admittime'] = X_test['admittime'] - X_test['admittime'].min()


    scaler = StandardScaler()
    X_train[['age', 'admittime']] = scaler.fit_transform(X_train[['age', 'admittime']])
    X_test[['age', 'admittime']] = scaler.transform(X_test[['age', 'admittime']])

    X_train.columns = X_train.columns.astype('str')
    X_test.columns = X_test.columns.astype('str')
    
    return X_train, X_test

def prep_signs_data():
    # CURRENTLY DROPS TIME COL
    train_signs = pd.read_csv('./train/train_signs.csv')
    test_signs = pd.read_csv('./test/test_signs.csv')

    train_signs['charttime'] = pd.to_datetime(train_signs['charttime'])
    test_signs['charttime'] = pd.to_datetime(test_signs['charttime'])
    
    train_signs['firsttime'] = train_signs['patient_id'].map(train_signs.groupby('patient_id')['charttime'].first())
    test_signs['firsttime'] = test_signs['patient_id'].map(test_signs.groupby('patient_id')['charttime'].first())
    # Sets the index as the time from the first reading so all patients start at 0 and go toward 24 hours
    train_signs['timediff'] = pd.to_numeric(train_signs['charttime'] - train_signs['firsttime'])
    test_signs['timediff'] = pd.to_numeric(test_signs['charttime'] - test_signs['firsttime'])

    train_signs = train_signs.drop(['charttime','firsttime'],axis=1)
    test_signs = test_signs.drop(['charttime','firsttime'],axis=1)
    
    X_train = train_signs.groupby('patient_id').agg(['mean', 'min', 'max', 'first', 'last'])
    X_test = test_signs.groupby('patient_id').agg(['mean', 'min', 'max', 'first', 'last'])

    scaler = StandardScaler()
    features = X_train.columns
    id = X_train.index
    id_test = X_test.index

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    X_train.columns = ['_'.join(x) for x in features]
    X_train.index = id
    X_test.columns = ['_'.join(x) for x in features]
    X_test.index = id_test

    # columns with more than 10% null values, drop these (10 columns, 2 metrics)
    drop_cols = X_train.columns[X_train.isna().sum() / X_train.shape[0] > .1] # should be just train set
    X_train = X_train.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)

    # is this the best way to do it??
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean()) # note train mean
    
    X_test.columns = X_test.columns.astype(str)
    X_train.columns = X_train.columns.astype(str)

    return X_train, X_test

def prep_radiology_data():
    X_train = pd.read_csv('./train/train_radiology.csv')
    X_test = pd.read_csv('./test/test_radiology.csv')

    X_train = X_train.groupby('patient_id').agg({'text': ['sum']})
    X_test = X_test.groupby('patient_id').agg({'text': ['sum']})

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=.5, min_df=5, max_features=100, stop_words="english")
    vec_train = vectorizer.fit_transform(X_train['text']['sum'])
    vec_test = vectorizer.transform(X_test['text']['sum'])
    X_train = pd.concat([X_train.drop(columns=['text']), pd.DataFrame(vec_train.toarray(), index=X_train.index)], axis=1)
    X_test = pd.concat([X_test.drop(columns=['text']), pd.DataFrame(vec_test.toarray(), index=X_test.index)], axis=1)

    scaler = StandardScaler()
    features = X_train.columns
    id = X_train.index
    id_test = X_test.index

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    X_train.columns = features
    X_train.index = id
    X_test.columns = features
    X_test.index = id_test

    X_test.columns = X_test.columns.astype(str)
    X_train.columns = X_train.columns.astype(str)

    return X_train, X_test