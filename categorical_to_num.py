import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import pickle


import logging
from logging_code import setup_logging
logger=setup_logging('categorical_to_num')

from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder


def c_t_n(x_train_cat,x_test_cat):
    try:
        logger.info(f'Before x_train_cat Columns {x_train_cat.shape} \n : {x_train_cat.columns}')
        logger.info(f'Befoer x_test_cat Columns {x_test_cat.shape} \n : {x_test_cat.columns}')
        oh=OneHotEncoder(drop='first')
        oh.fit(x_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines','InternetService', 'OnlineSecurity',
                            'OnlineBackup', 'DeviceProtection','TechSupport', 'StreamingTV', 'StreamingMovies','PaperlessBilling',
                            'PaymentMethod', 'Networks']])
        value_train=oh.transform(x_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines','InternetService',
                                              'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport', 'StreamingTV',
                                              'StreamingMovies','PaperlessBilling', 'PaymentMethod', 'Networks']]).toarray()
        value_test=oh.transform(x_test_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines','InternetService',
                                            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport', 'StreamingTV',
                                            'StreamingMovies','PaperlessBilling', 'PaymentMethod', 'Networks']]).toarray()
        t1=pd.DataFrame(value_train)
        t2=pd.DataFrame(value_test)
        t1.columns=oh.get_feature_names_out()
        t2.columns=oh.get_feature_names_out()
        x_train_cat.reset_index(drop=True, inplace=True)
        x_test_cat.reset_index(drop=True, inplace=True)
        t1.reset_index(drop=True, inplace=True)
        t2.reset_index(drop=True, inplace=True)
        x_train_cat=pd.concat([x_train_cat,t1],axis=1)
        x_test_cat=pd.concat([x_test_cat,t2],axis=1)
        x_train_cat=x_train_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines','InternetService',
                                      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport', 'StreamingTV', 'StreamingMovies',
                                      'PaperlessBilling', 'PaymentMethod', 'Networks'],axis=1)
        x_test_cat=x_test_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines','InternetService',
                                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport', 'StreamingTV', 'StreamingMovies',
                                    'PaperlessBilling', 'PaymentMethod', 'Networks'],axis=1)
        logger.info(f'After NOMINAL x_train_cat Columns {x_train_cat.shape} \n : {x_train_cat.columns}')
        logger.info(f'After NOMINAL x_test_cat Columns {x_test_cat.shape} \n : {x_test_cat.columns}')

        logger.info(f'===========================OrdinalEncoder=============================================================')

        od=OrdinalEncoder()
        od.fit(x_train_cat[['Contract']])
        results_train=od.transform(x_train_cat[['Contract']])
        results_test=od.transform(x_test_cat[['Contract']])
        p1=pd.DataFrame(results_train)
        p2=pd.DataFrame(results_test)
        p1.columns=od.get_feature_names_out()+'_od'
        p2.columns=od.get_feature_names_out()+'_od'
        p1.reset_index(drop=True,inplace=True)
        p2.reset_index(drop=True,inplace=True)
        x_train_cat=pd.concat([x_train_cat,p1],axis=1)
        x_test_cat=pd.concat([x_test_cat,p2],axis=1)
        x_train_cat=x_train_cat.drop(['Contract'],axis=1)
        x_test_cat=x_test_cat.drop(['Contract'],axis=1)

        logger.info(f'After ODINAL x_train_cat Columns {x_train_cat.shape} \n : {x_train_cat.columns}')
        logger.info(f'After ODINAL x_test_cat Columns {x_test_cat.shape} \n : {x_test_cat.columns}')

        logger.info(f'Train NULL VALUES : \n{x_train_cat.isnull().sum()}')
        logger.info(f'Test NULL VALUES : \n{x_test_cat.isnull().sum()}')


        return  x_train_cat,x_test_cat




    except Exception as e:
        msg, typ, lin = sys.exc_info()
        logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')