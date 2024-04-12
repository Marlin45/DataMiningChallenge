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

def train_NN(X_train,y_train):
    """
        Trains a neural network with hyperparamters determined in jupyter notebook
        Also saves the model in model.keras so it may be loaded instead of running this function
    """
    input_size = X_train.shape[1]

    # Three layer network using regularization and dropout to reduce overfitting, with a one neuron sigmoid activation function to give the probability
    model = keras.Sequential([
            layers.Dense(input_size, activation="tanh", name='Input-Layer'),
            layers.Dense(units=96,activation="tanh", kernel_regularizer=keras.regularizers.L1L2(l1=1e-2, l2=1e-2),name='Hidden-Layer-1'),
            layers.Dropout(0.5),
            layers.Dense(units=1,activation='sigmoid',name='Output')
    ])

    # Reduces the learning rate twice druing training for better convergence
    for lr in [1e-4,1e-5,1e-6]:
        opt = keras.optimizers.Adam(learning_rate=lr)
        callbacks = keras.callbacks.EarlyStopping(monitor='val_loss',patience=5) 
        model.compile(optimizer=opt, loss=keras.losses.BinaryCrossentropy(),metrics=[keras.metrics.AUC()])
        model.fit(X_train,y_train,validation_split=0.1,epochs=1000,callbacks=[callbacks],verbose=1)
    model.save("model.keras")
    return model