import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import yeojohnson
from scipy.stats import median_abs_deviation

import logging
from logging_code import setup_logging
logger=setup_logging('var_out')

def vt_outliers(x_train_num,x_test_num):
    try:
        # for i in x_train_num.columns:
        #     plt.figure(figsize=(8, 3))
        #     plt.subplot(1, 2, 1)
        #     plt.title('Normal Distribution')
        #     x_train_num[i].plot(kind='kde')
        #
        #     plt.subplot(1, 2, 2)
        #     plt.title('Outliers')
        #     sns.boxplot(x=x_train_num[i])
        #     plt.show()
        logger.info(f'Before Train columns names : {x_train_num.columns}')
        logger.info(f'Before test columns names : {x_test_num.columns}')
        # yeojohnson
        p=x_train_num.drop(['SeniorCitizen'],axis=1)   # in SeniorCitizen binary data
        # print(p.columns)
        for i in p.columns:
            x_train_num[i + '_yeo'], lam_value = yeojohnson((x_train_num[i]))
            x_test_num[i + '_yeo'], lam_value = yeojohnson((x_test_num[i]))
            x_train_num = x_train_num.drop([i], axis=1)
            x_test_num = x_test_num.drop([i], axis=1)
        # Median Absolute Deviation (MAD Method)
            median = x_train_num[i + '_yeo'].median()
            mad = median_abs_deviation(x_train_num[i + '_yeo'])
            lower_limit = median - 3 * mad
            upper_limit = median + 3 * mad
            x_train_num[i + '_mad'] = np.where(x_train_num[i + '_yeo'] > upper_limit, upper_limit,
                             np.where(x_train_num[i + '_yeo'] < lower_limit, lower_limit, x_train_num[i + '_yeo']))
            x_test_num[i + '_mad'] = np.where(x_test_num[i + '_yeo'] > upper_limit, upper_limit,
                                      np.where(x_test_num[i + '_yeo'] < lower_limit, lower_limit, x_test_num[i + '_yeo']))
            x_train_num = x_train_num.drop([i + '_yeo'], axis=1)
            x_test_num = x_test_num.drop([i + '_yeo'], axis=1)
        logger.info(f'After Train columns names : {x_train_num.columns}')
        logger.info(f'After test columns names : {x_test_num.columns}')

        return x_train_num, x_test_num

    except Exception as e:
        msg, typ, lin = sys.exc_info()
        logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')
