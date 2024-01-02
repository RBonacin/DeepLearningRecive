# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn 
import math
import shap
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split #Splitting data into training and testing sets
from imblearn.metrics import specificity_score
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras import layers
from keras import layers, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve


# Create a LSTM  Model
def create_LSTM_model():
    model = Sequential()
    model.add(layers.Embedding(input_dim=230, output_dim=64,input_length=23))   # Dim Tested from 128 - 230
    model.add(
        layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(5, 10))
    )
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dense(10))
    model.add(Dense(1, activation='sigmoid'))
    #Compiling model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC()])
    return model


# Initialize and read the dataset
def init():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    file = r'reduced_onco_dec_2019.csv'
    df = pd.read_csv(file)
    return df


#Execute Split 70 - 30 for each ICD
def splitICD(filtered_df):
    x = filtered_df[[
        "idade","sexo","ibge","basediag","topo","morfo","ec","t","n","m","psa","gleason","meta01","meta02","meta03",
        "meta04","naotrat","trathosp","tratfapos","consdiag","tratcons","diagtrat","cici"]].values #Seja x a matrix com as features que quero analisar
    y = filtered_df["recnenhum"].values #Seja y o vetor que indica se houve ou não reincidiva
    oversample = SMOTE()
    model=create_LSTM_model()
    train_data, test_data, train_label, test_label = train_test_split(x,y, test_size=0.3, stratify=None)
    train_data, train_label = oversample.fit_resample(train_data, train_label)
    model.fit(train_data, train_label, epochs=20)
    #explainer = shap.KernelExplainer(model.predict,train_data)
    predictions = model.predict(test_data)
    pred = np.round_(predictions)
    #shap_values = explainer.shap_values(test_data,nsamples=20)
    FEATURES = ["idade","sexo","ibge","basediag","topo","morfo","ec","t","n","m","psa","gleason","meta01","meta02","meta03",
        "meta04","naotrat","trathosp","tratfapos","consdiag","tratcons","diagtrat","cici"]

    print('<<<<< SHAP GRAPH HERE >>>>')    
    #plt.clf()
    #shap.summary_plot(shap_values,test_data,feature_names=FEATURES,show=False)
    #plt.savefig("summary_plot.png",dpi=300, bbox_inches='tight')
    explainer = shap.Explainer(model, train_data)
    explainer.feature_names = FEATURES
    shap_values = explainer(train_data)
    plt.clf()
    shap.plots.beeswarm(shap_values,show=False)
    plt.savefig("beeswarmLSTM.png",dpi=300, bbox_inches='tight')
    f1 = sklearn.metrics.f1_score(test_label,pred)
    print('> Split 70-30 ')
    print('F1Score: ',f1)
    mcc = sklearn.metrics.matthews_corrcoef(test_label,pred)
    print('MCC: ', mcc)
    spe = specificity_score(test_label,pred,average='weighted')
    print('Specificity: ', spe)
    acc = sklearn.metrics.accuracy_score(test_label, pred)
    print('Accuracy: ', acc)
    auc = roc_auc_score(test_label, pred)
    print('AUC : ',auc)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    res = tf.math.confusion_matrix(test_label,pred, dtype=tf.dtypes.int32)
    print('Confusion_matrix Split 70-30 : ',res)




# Main Method - All ICD in the Same
df = init()
#query_topo = "topo == " + str(715) 
#filtered_df = df.query(query_topo)
filtered_df = df
print("Size for ALL ICDs is " + str(filtered_df.size/24))
splitICD(filtered_df)  # Split for ICD
  
  








