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
logger=setup_logging('filter_methods')

from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr


def fm(x_train_num,x_test_num,y_train,y_test):
    try:
        logger.info(f'Before train columns : {x_train_num.shape} \n :  {x_train_num.columns}')
        logger.info(f'Before test columns : {x_test_num.shape} \n : {x_test_num.columns}')
        reg = VarianceThreshold(threshold=0.01)
        reg.fit(x_train_num)
        logger.info(f'Number of GOOD COLUMNS : {sum(reg.get_support())} : {x_train_num.columns[reg.get_support()]}')
        logger.info(f'Number of BAD COLUMNS : {sum(~reg.get_support())} : {x_train_num.columns[~reg.get_support()]}')
        logger.info(f'After FILTER METHODS train columns : {x_train_num.shape} \n :  {x_train_num.columns}')
        logger.info(f'After FILTER METHODS test columns : {x_test_num.shape} \n : {x_test_num.columns}')

        logger.info(f'===========================HYPOTHESIS_TESTING=============================================================')
        c = []
        for i in x_train_num.columns:
            r = pearsonr(x_train_num[i], y_train)
            c.append(r)
        t = np.array(c)
        p_value = pd.Series(t[:, 1], index=x_train_num.columns)
        # p = 0
        # f = []
        # for i in p_value:
        #     if i < 0.05:
        #         f.append(x_train_num.columns[p])
        #     p = p + 1
        # print(x_train_num.columns)
        # print(f) # good columns
        logger.info(f'After HYPOTHESIS TESTING train columns : {x_train_num.shape} \n :  {x_train_num.columns}')
        logger.info(f'After HYPOTHESIS TESTING test columns : {x_test_num.shape} \n : {x_test_num.columns}')
        return x_train_num, x_test_num
    except Exception as e:
        msg, typ, lin = sys.exc_info()
        logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')