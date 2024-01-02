# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn 
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
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
    with open('outputALLCIDLSTM.txt', 'w') as log:
        log.close()
    return df


#Execute Stratified CrossValidation for imbalanced datasets for each ICD
def StratifiedCrossCID(filtered_df):
    x = filtered_df[[
        "idade","sexo","ibge","basediag","topo","morfo","ec","t","n","m","psa","gleason","meta01","meta02","meta03",
        "meta04","naotrat","trathosp","tratfapos","consdiag","tratcons","diagtrat","cici"]].values #Seja x a matrix com as features que quero analisar
    y = filtered_df["recnenhum"].values #Seja y o vetor que indica se houve ou não reincidiva
    oversample = SMOTE()
    n_split=5
    sf1 = 0
    sacc = 0
    smcc = 0
    sspe = 0
    sauc = 0
    for train_index,test_index in StratifiedKFold(n_split,random_state=25, shuffle=True).split(x,y):
      x_train,x_test=x[train_index],x[test_index]
      y_train,y_test=y[train_index],y[test_index]
      # Oversampling the traning data
      x_train, y_train = oversample.fit_resample(x_train, y_train)
      model=create_LSTM_model()
      model.fit(x_train, y_train,epochs=30)
      predictions = model.predict(x_test)
      pred = np.round_(predictions)
      f1 = sklearn.metrics.f1_score(y_test,pred)
      print('F1Score: ',f1)
      sf1 = sf1 + f1
      mcc = sklearn.metrics.matthews_corrcoef(y_test,pred)
      print('MCC: ', mcc)
      smcc = smcc + mcc
      spe = specificity_score(y_test,pred,average='weighted')
      print('Specificity: ', spe)
      sspe = sspe + spe  
      auc = roc_auc_score(y_test, pred)
      acc = sklearn.metrics.accuracy_score(y_test, pred)
      print('Accuracy: ', acc)
      sacc = sacc + acc
      print('AUC : ',auc)
      sauc = sauc + auc
      print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
      with open('outputALLCIDLSTM.txt', 'a') as log:
          log.write(str(f1))
          log.write(',')
          log.write(str(mcc))
          log.write(',')
          log.write(str(spe))
          log.write(',')
          log.write(str(acc))
          log.write(',')
          log.write(str(auc))
          log.write(',')
          log.write('CrossVal')
          log.write('\n')
          log.close()

     
      
    # CrossValidation Results
    print('> Stratified CrossValidation (avarage) Results');   
    print('F1Score CrossValidation Mean = ', sf1/n_split)
    print('MCC CrossValidation Mean = ', smcc/n_split)
    print('SPE CrossValidation Mean = ', sspe/n_split)
    print('Accuracy CrossValidation Mean = ', sacc/n_split)
    print('AUC CrossValidation Mean = ', sauc/n_split)
    with open('outputALLCIDLSTM.txt', 'a') as log:
        log.write(str(sf1/n_split))
        log.write(',')
        log.write(str(smcc/n_split))
        log.write(',')
        log.write(str(sspe/n_split))
        log.write(',')
        log.write(str(sacc/n_split))
        log.write(',')
        log.write(str(sauc/n_split))
        log.write(',')
        log.write('CrossValAVG')
        log.write('\n')
        log.close()

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
    model.fit(train_data, train_label, epochs=30)
    predictions = model.predict(test_data)
    pred = np.round_(predictions)
    f1 = sklearn.metrics.f1_score(test_label,pred)
    print('> Split 70-30 ')
    print('F1Score: ',f1)
    with open('outputALLCIDLSTM.txt', 'a') as log:
        log.write(str(f1))
        log.write(',')
        log.close()
    mcc = sklearn.metrics.matthews_corrcoef(test_label,pred)
    print('MCC: ', mcc)
    with open('outputALLCIDLSTM.txt', 'a') as log:
        log.write(str(mcc))
        log.write(',')
        log.close()
    spe = specificity_score(test_label,pred,average='weighted')
    print('Specificity: ', spe)
    with open('outputALLCIDLSTM.txt', 'a') as log:
        log.write(str(spe))
        log.write(',')
        log.close()
    acc = sklearn.metrics.accuracy_score(test_label, pred)
    print('Accuracy: ', acc)
    with open('outputALLCIDLSTM.txt', 'a') as log:
        log.write(str(acc))
        log.write(',')
        log.close()
    auc = roc_auc_score(test_label, pred)
    print('AUC : ',auc)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    with open('outputALLCIDLSTM.txt', 'a') as log:
        log.write(str(auc))
        log.write(',')
        log.close()
    with open('outputALLCIDLSTM.txt', 'a') as log:
        log.write('> Split 70-30 ')
        log.write('\n')
        log.close()
            
    res = tf.math.confusion_matrix(test_label,pred, dtype=tf.dtypes.int32)
    print('Confusion_matrix Split 70-30 : ',res)




# Main Method - All ICD in the Same
df = init()
filtered_df = df   # ALL ICDs
print("Size for ALL ICDs is " + str(filtered_df.size/24))
with open('outputALLCIDLSTM.txt', 'a') as log:
    log.write('F1Score,MCC,Specificity,Accuracy,AUC,Phase')
    log.write('\n')
    log.close()
StratifiedCrossCID(filtered_df)
splitICD(filtered_df)  # Split for ICD
  
  








