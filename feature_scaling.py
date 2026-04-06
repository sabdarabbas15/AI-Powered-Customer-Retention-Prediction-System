import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

import logging
from logging_code import setup_logging
logger=setup_logging('feature_scaling')

from sklearn.preprocessing import Normalizer # z score

from all_models import common
from all_models import hypertuning

from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

import pickle

def fs(x_train,y_train,x_test,y_test):
    try:
        logger.info(f'Training data independent size : {x_train.shape}')
        logger.info(f'Training data dependent size : {y_train.shape}')
        logger.info(f'testing data independent size : {x_test.shape}')
        logger.info(f'testing data dependent size : {y_test.shape}')
        logger.info(f'before {x_train.head(1)}')

        scaler = Normalizer(norm='l2')  # try 'l1' also
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        logger.info(f'{x_train_scaled}')

        with open('Normalizer.pkl','wb') as f:
            pickle.dump(scaler,f)

        #common(x_train_scaled,y_train,x_test_scaled,y_test)

        lr_reg = LogisticRegression(C= 10, class_weight= 'balanced', max_iter= 500, penalty= 'l1', solver= 'liblinear')
        lr_reg.fit(x_train_scaled, y_train)
        lr_pred = lr_reg.predict(x_test_scaled)

        logger.info(confusion_matrix(y_test, lr_pred))
        logger.info(accuracy_score(y_test, lr_pred))
        logger.info(classification_report(y_test, lr_pred))

        with open('model.pkl', 'wb') as t:
            pickle.dump(lr_reg, t)

    except Exception as e:
        msg, typ, lin = sys.exc_info()
        logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')