import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import warnings
warnings.filterwarnings("ignore")

import logging
from logging_code import  setup_logging
logger = setup_logging('mode')

def handling_missing_value(x_train,x_test):
    try:
        logger.info(f'before handling missing value x_train shape and columns : {x_train.shape} \n : {x_train.columns} : {x_train.isnull().sum()}')
        logger.info(f'before handling missing value x_test shape and columns : {x_test.shape} \n : {x_test.columns} : {x_test.isnull().sum()}')
        for i in x_train.columns:
            if x_train[i].isnull().sum() > 0:
                x_train[i+'_mode'] = x_train[i].fillna(x_train[i].mode()[0])
                x_test[i+'_mode'] = x_test[i].fillna(x_test[i].mode()[0])
                x_train = x_train.drop([i], axis=1)
                x_test = x_test.drop([i], axis=1)
        logger.info(f'After handling missing value x_train shape and columns : {x_train.shape} \n : {x_train.columns}  : {x_train.isnull().sum()}')
        logger.info(f'After handling missing value x_test shape and columns : {x_test.shape} \n : {x_test.columns} : {x_test.isnull().sum()}')
        return  x_train,x_test
    except Exception as e:
        msg, typ, lin = sys.exc_info()
        logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')