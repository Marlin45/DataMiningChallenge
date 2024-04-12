import pandas as pd
import numpy as np
from sklearn import svm
import keras
from scripts import prep_data,nn

RANDOM_STATE = 2405023

if __name__ == "__main__":
    
    # Reads and preprocess training and testing  data
    demo_train, demo_test = prep_data.prep_demo_data()
    signs_train, signs_test = prep_data.prep_signs_data()
    rad_train, rad_test = prep_data.prep_radiology_data()

    # Prepares train, test, and lable dataframes
    X_train = pd.concat([demo_train,signs_train,rad_train],axis=1)
    X_test = pd.concat([demo_test,signs_test,rad_test],axis=1)
    y_train = pd.read_csv("./train/train_labels.csv")
    y_train.set_index('patient_id',inplace=True)

    # Balances the training data using random undersampling (RUS)
    keep_ids = [*y_train[y_train['label'] == 0].sample(n=y_train['label'].sum(),random_state=RANDOM_STATE).index,*y_train[y_train['label'] == 1].index]
    y_train = y_train.loc[keep_ids]
    X_train = X_train.loc[keep_ids]

    # Fit and predicts SVM
    svc = svm.SVC(probability=True,kernel='rbf')
    fit_svc = svc.fit(X_train,np.array(y_train).ravel())
    svm_predictions = svc.predict_proba(X_test)[:, 1].reshape(-1,1)

    # Either trains or loads the neural network and predicts
    # Uncomment this line to train the model, otherwise it will load the model trained here from the model.keras file
    # model = nn.train_NN(X_train,y_train)
    model = keras.models.load_model('model.keras')
    nn_predictions = model.predict(X_test)

    # Produces ensemble estimate with simple Committee of models just takes the mean prob
    predictions = (svm_predictions + nn_predictions) / 2

    # Writes predictions to csv
    predictions = pd.DataFrame(predictions,index=X_test.index,columns=['probability'])
    predictions.to_csv("predictions.csv")