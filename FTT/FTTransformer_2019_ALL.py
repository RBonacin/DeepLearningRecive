#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 17:07:23 2023

@author: rodrigo
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import sklearn
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tabtransformertf.utils.preprocessing import df_to_dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.metrics import specificity_score
from sklearn.preprocessing import StandardScaler
from tabtransformertf.models.fttransformer import FTTransformerEncoder, FTTransformer
from tabtransformertf.utils.preprocessing import df_to_dataset, build_categorical_prep
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import gc


# Initialize and read the dataset
def init():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    file = r'reduced_onco_dec_2019.csv'
    df = pd.read_csv(file)
    for c in df.columns:
        df[c] = df[c].astype(float)

    with open('outputALLCIDFTT_ALL.txt', 'w') as log:
        log.close()
    return df


#Execute Stratified CrossValidation for imbalanced datasets for each ICD
def StratifiedCrossCID(filtered_df,topo,size):
    NUMERIC_FEATURES = ["idade","consdiag","tratcons","diagtrat"]
    
    CATEGORICAL_FEATURES = ["sexo","ibge","basediag","topo","morfo","ec","t","n","m","psa","gleason","meta01","meta02","meta03",
    "meta04","naotrat","trathosp","tratfapos","cici"]
    
    FEATURES = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    
    
    LABEL = 'recnenhum'
    
    x = filtered_df[FEATURES].values #Seja x a matrix com as features que quero analisar
    y = filtered_df[LABEL].values #Seja y o vetor que indica se houve ou não reincidiva
    
    oversample = SMOTE()
    n_split=5
    sf1 = 0
    sacc = 0
    smcc = 0
    sspe = 0
    sauc = 0
    for train_index,test_index in StratifiedKFold(n_split,random_state=25, shuffle=True).split(x,y):
        gc.collect()
        #print(train_index)
        #print(test_index)
        X_train = filtered_df.iloc[train_index]
        X_val = filtered_df.iloc[test_index]
        
        # Oversampling the traning set
        oversample = SMOTE()
        X_train,yy_train = oversample.fit_resample(X_train[FEATURES],X_train[LABEL])
        yy_train = yy_train.to_frame()
        X_train = pd.concat([X_train, yy_train], axis=1)
        
        # Oversampling the traning set
        oversample = SMOTE()
        X_train,yy_train = oversample.fit_resample(X_train[FEATURES],X_train[LABEL])
        yy_train = yy_train.to_frame()
        X_train = pd.concat([X_train, yy_train], axis=1)
        
        # Set data types
        #train_data[CATEGORICAL_FEATURES] = train_data[CATEGORICAL_FEATURES].astype(str)
        #test_data[CATEGORICAL_FEATURES] = test_data[CATEGORICAL_FEATURES].astype(str)
        X_train[CATEGORICAL_FEATURES] = X_train[CATEGORICAL_FEATURES].astype(str)
        X_val[CATEGORICAL_FEATURES] = X_val[CATEGORICAL_FEATURES].astype(str)
        
        #train_data[NUMERIC_FEATURES] = train_data[NUMERIC_FEATURES].astype(float)
        #test_data[NUMERIC_FEATURES] = test_data[NUMERIC_FEATURES].astype(float)
        X_train[NUMERIC_FEATURES] = X_train[NUMERIC_FEATURES].astype(float)
        X_val[NUMERIC_FEATURES] = X_val[NUMERIC_FEATURES].astype(float)
        
        sc = StandardScaler()
        X_train.loc[:, NUMERIC_FEATURES] = sc.fit_transform(X_train[NUMERIC_FEATURES])
        X_val.loc[:, NUMERIC_FEATURES] = sc.transform(X_val[NUMERIC_FEATURES])
        #test_data.loc[:, NUMERIC_FEATURES] = sc.transform(test_data[NUMERIC_FEATURES])

        train_dataset = df_to_dataset(X_train[FEATURES + [LABEL]], LABEL)
        val_dataset = df_to_dataset(X_val[FEATURES + [LABEL]], LABEL, shuffle=False)  # No shuffle
        #test_dataset = df_to_dataset(test_data[FEATURES + [LABEL]], shuffle=False) # No target, no shuffle
        ft_linear_encoder = FTTransformerEncoder(
            numerical_features = NUMERIC_FEATURES,
            categorical_features = CATEGORICAL_FEATURES,
            numerical_data = X_train[NUMERIC_FEATURES].values,
            categorical_data = X_train[CATEGORICAL_FEATURES].values,
            y = None,
            numerical_embedding_type='linear',
            embedding_dim=16,
            depth=4,
            heads=8,
            attn_dropout=0.2,
            ff_dropout=0.2,
            explainable=True
        )            

        
        # Pass the encoder to the model
        ft_linear_transformer = FTTransformer(
            encoder=ft_linear_encoder,
            out_dim=1,
            out_activation='sigmoid',
        )
        
        LEARNING_RATE = 0.001
        WEIGHT_DECAY = 0.0001
        NUM_EPOCHS = 3

        optimizer = tfa.optimizers.AdamW(
            learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
    
        ft_linear_transformer.compile(
            optimizer = optimizer,
            loss = {"output": tf.keras.losses.BinaryCrossentropy(), "importances": None},
            metrics= {"output": [tf.keras.metrics.AUC(name="PR AUC", curve='PR')], "importances": None},
        )


        early = EarlyStopping(monitor="val_output_loss", mode="min", patience=20, restore_best_weights=True)
        callback_list = [early]
        
        ft_linear_history = ft_linear_transformer.fit(
            train_dataset, 
            epochs=NUM_EPOCHS, 
            validation_data=val_dataset,
            callbacks=callback_list
        )

        val_dataset = df_to_dataset(X_val[FEATURES + [LABEL]], LABEL, shuffle=False)  # No shuffle
        
        linear_test_preds = ft_linear_transformer.predict(val_dataset)
        pred = linear_test_preds['output'].ravel()>=0.5
        y_test = X_val[LABEL]
               
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
        with open('outputALLCIDFTT_ALL.txt', 'a') as log:
            log.write(str(topo))
            log.write(',')
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
            log.write(',')
            log.write(str(size))
            log.write('\n')
            log.close()
      
    # CrossValidation Results
    print('> Stratified CrossValidation (avarage) Results');   
    print('F1Score CrossValidation Mean = ', sf1/n_split)
    print('MCC CrossValidation Mean = ', smcc/n_split)
    print('SPE CrossValidation Mean = ', sspe/n_split)
    print('Accuracy CrossValidation Mean = ', sacc/n_split)
    print('AUC CrossValidation Mean = ', sauc/n_split)
    with open('outputALLCIDFTT_ALL.txt', 'a') as log:
        log.write(str(topo))
        log.write(',')
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
        log.write(',')
        log.write(str(size))
        log.write('\n')
        log.close()

#Execute Split 70 - 30 for each ICD
def splitICD(filtered_df,topo,size):
    NUMERIC_FEATURES = ["idade","consdiag","tratcons","diagtrat"]
    
    CATEGORICAL_FEATURES = ["sexo","ibge","basediag","topo","morfo","ec","t","n","m","psa","gleason","meta01","meta02","meta03",
    "meta04","naotrat","trathosp","tratfapos","cici"]
    
    FEATURES = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    
    
    LABEL = 'recnenhum'
    
    
    # Train/test split (80 - 20)
    train_data, test_data = train_test_split(filtered_df, test_size=0.3)
    
    
    
    X_train, X_val = train_test_split(train_data, test_size=0.2)
    
    
    # Oversampling the traning set
    oversample = SMOTE()
    X_train,yy_train = oversample.fit_resample(X_train[FEATURES],X_train[LABEL])
    yy_train = yy_train.to_frame()
    X_train = pd.concat([X_train, yy_train], axis=1)
    
    # Set data types
    #train_data[CATEGORICAL_FEATURES] = train_data[CATEGORICAL_FEATURES].astype(str)
    test_data[CATEGORICAL_FEATURES] = test_data[CATEGORICAL_FEATURES].astype(str)
    X_train[CATEGORICAL_FEATURES] = X_train[CATEGORICAL_FEATURES].astype(str)
    X_val[CATEGORICAL_FEATURES] = X_val[CATEGORICAL_FEATURES].astype(str)
    
    #train_data[NUMERIC_FEATURES] = train_data[NUMERIC_FEATURES].astype(float)
    test_data[NUMERIC_FEATURES] = test_data[NUMERIC_FEATURES].astype(float)
    X_train[NUMERIC_FEATURES] = X_train[NUMERIC_FEATURES].astype(float)
    X_val[NUMERIC_FEATURES] = X_val[NUMERIC_FEATURES].astype(float)
    
    
    
    
    sc = StandardScaler()
    X_train.loc[:, NUMERIC_FEATURES] = sc.fit_transform(X_train[NUMERIC_FEATURES])
    X_val.loc[:, NUMERIC_FEATURES] = sc.transform(X_val[NUMERIC_FEATURES])
    test_data.loc[:, NUMERIC_FEATURES] = sc.transform(test_data[NUMERIC_FEATURES])
    
    
    train_dataset = df_to_dataset(X_train[FEATURES + [LABEL]], LABEL)
    val_dataset = df_to_dataset(X_val[FEATURES + [LABEL]], LABEL, shuffle=False)  # No shuffle
    test_dataset = df_to_dataset(test_data[FEATURES + [LABEL]], shuffle=False) # No target, no shuffle
    
    ft_linear_encoder = FTTransformerEncoder(
        numerical_features = NUMERIC_FEATURES,
        categorical_features = CATEGORICAL_FEATURES,
        numerical_data = X_train[NUMERIC_FEATURES].values,
        categorical_data = X_train[CATEGORICAL_FEATURES].values,
        y = None,
        numerical_embedding_type='linear',
        embedding_dim=16,
        depth=4,
        heads=8,
        attn_dropout=0.2,
        ff_dropout=0.2,
        explainable=True
    )
    
    # Pass the encoder to the model
    ft_linear_transformer = FTTransformer(
        encoder=ft_linear_encoder,
        out_dim=1,
        out_activation='sigmoid',
    )
    
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    NUM_EPOCHS = 3
    
    optimizer = tfa.optimizers.AdamW(
            learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    
    ft_linear_transformer.compile(
        optimizer = optimizer,
        loss = {"output": tf.keras.losses.BinaryCrossentropy(), "importances": None},
        metrics= {"output": [tf.keras.metrics.AUC(name="PR AUC", curve='PR')], "importances": None},
    )
    
    early = EarlyStopping(monitor="val_output_loss", mode="min", patience=20, restore_best_weights=True)
    callback_list = [early]
    
    ft_linear_history = ft_linear_transformer.fit(
        train_dataset, 
        epochs=NUM_EPOCHS, 
        validation_data=val_dataset,
        callbacks=callback_list
    )
       
    linear_test_preds = ft_linear_transformer.predict(test_dataset)
    pred = linear_test_preds['output'].ravel()>=0.5
    test_label = test_data[LABEL]
    print('> Split 70-30 ')
    f1 = sklearn.metrics.f1_score(test_label,pred)
    print('F1Score: ',f1)
    with open('outputALLCIDFTT_ALL.txt', 'a') as log:
        log.write(str(topo))
        log.write(',')
        log.write(str(f1))
        log.write(',')
        log.close()
    mcc = sklearn.metrics.matthews_corrcoef(test_label,pred)
    print('MCC: ', mcc)
    with open('outputALLCIDFTT_ALL.txt', 'a') as log:
        log.write(str(mcc))
        log.write(',')
        log.close()
    spe = specificity_score(test_label,pred,average='weighted')
    print('Specificity: ', spe)
    with open('outputALLCIDFTT_ALL.txt', 'a') as log:
        log.write(str(spe))
        log.write(',')
        log.close()
    acc = sklearn.metrics.accuracy_score(test_label, pred)
    print('Accuracy: ', acc)
    with open('outputALLCIDFTT_ALL.txt', 'a') as log:
        log.write(str(acc))
        log.write(',')
        log.close()
    auc = roc_auc_score(test_label, pred)
    print('AUC : ',auc)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    with open('outputALLCIDFTT_ALL.txt', 'a') as log:
        log.write(str(auc))
        log.write(',')
        log.close()
    with open('outputALLCIDFTT_ALL.txt', 'a') as log:
        log.write('Split')
        log.write(',')
        log.write(str(size))
        log.write('\n')
        log.close()

# Main Method 
df = init()
with open('outputALLCIDFTT_ALL.txt', 'a') as log:
    log.write('CID,F1Score,MCC,Specificity,Accuracy,AUC,Phase,size')
    log.write('\n')
    log.close() 
    i = 0
#for topo in df['topo'].unique():   # For Each ICD
#    query_topo = "topo == " + str(topo) 
filtered_df = df
size = filtered_df.size/24
topo = "topo"
i = i + 1
gc.collect(generation=1)
print("i======== " + str(i))
print("Size for ICD " + str(topo) + " is " + str(size))
if (size>200) :  # Ignore ICD with less than 200 instances
    StratifiedCrossCID(filtered_df,topo,size)
    splitICD(filtered_df,topo,size)  # Split for ICD
    gc.collect()
  
