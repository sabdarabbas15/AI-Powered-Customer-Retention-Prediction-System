import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sns

import logging
from logging_code import setup_logging
logger=setup_logging('main')

from mode import handling_missing_value
from var_out import vt_outliers
from filter_methods import fm
from categorical_to_num import c_t_n

from imblearn.over_sampling import SMOTE

from feature_scaling import fs


class CHURN:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            logger.info(f'total data size : {self.df.shape}')
            logger.info(f'null values : \n : {self.df.isnull().sum()}')
            self.df['TotalCharges']=self.df['TotalCharges'].replace(' ',np.nan)
            self.df['TotalCharges']=pd.to_numeric(self.df['TotalCharges'])

            np.random.seed(42)   # Same result every time
            networks = ['Airtel', 'BSNL', 'Jio', 'Idea']
            self.df['Networks'] = np.random.choice(networks, size=len(self.df))  # np.random.choice() randomly picks a value from the list
                                                                                # size=len(self.df) ensures every row gets a random network
            self.df=self.df.drop(['customerID'],axis=1)
            logger.info(self.df.info())
            logger.info(f'total data size : {self.df.shape}')
            logger.info(f'null values : \n{self.df.isnull().sum()}')
            self.x = self.df.drop(['Churn'],axis=1) # independent
            self.y = self.df['Churn']  # dependent
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2,random_state=42)
            self.y_train = self.y_train.map({'Yes': 1, 'No': 0}).astype(int)
            self.y_test = self.y_test.map({'Yes': 1, 'No': 0}).astype(int)
        except Exception as e:
            msg,typ,lin=sys.exc_info()
            logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')

    def missing_values(self):
        try:
            logger.info(f'before handling missing value x_train shape and columns : {self.x_train.shape} \n : {self.x_train.columns}  : {self.x_train.isnull().sum()}')
            logger.info(f'before handling missing value x_test shape and columns : {self.x_test.shape} \n : {self.x_test.columns} : {self.x_test.isnull().sum()}')
            self.x_train, self.x_test = handling_missing_value(self.x_train, self.x_test)
            logger.info(f'After handling missing value x_train shape and columns : {self.x_train.shape} \n : {self.x_train.columns}  : {self.x_train.isnull().sum()}')
            logger.info(f'After handling missing value x_test shape and columns : {self.x_test.shape} \n : {self.x_test.columns} : {self.x_test.isnull().sum()}')
        except Exception as e:
            msg,typ,lin=sys.exc_info()
            logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')

    def data_seperation(self):
        try:
            self.x_train_num_col=self.x_train.select_dtypes(exclude='object')
            self.x_test_num_col=self.x_test.select_dtypes(exclude='object')
            self.x_train_cat_col=self.x_train.select_dtypes(include='object')
            self.x_test_cat_col=self.x_test.select_dtypes(include='object')

            for i in self.x_train_num_col:
                logger.info(f'{i}  (DATA TYPE) : {self.x_train_num_col[i].dtype}')

            logger.info(f'{self.x_train_num_col.columns} : {self.x_train_num_col.shape}')
            logger.info(f'{self.x_test_num_col.columns} : {self.x_test_num_col.shape}')
            logger.info(f'================================================================')
            logger.info(f'{self.x_train_cat_col.columns} : {self.x_train_cat_col.shape}')
            logger.info(f'{self.x_test_cat_col.columns} : {self.x_test_cat_col.shape}')
        except Exception as e:
            msg,typ,lin=sys.exc_info()
            logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')

    def variable_transformation(self):
        try:
            logger.info(f'Before Train columns names : {self.x_train_num_col.columns}')
            logger.info(f'Before test columns names : {self.x_test_num_col.columns}')

            self.x_train_num_col,self.x_test_num_col = vt_outliers(self.x_train_num_col,self.x_test_num_col)

            logger.info(f'After Train columns names : {self.x_train_num_col.columns}')
            logger.info(f'After test columns names : {self.x_test_num_col.columns}')
            #self.x_train_num_col.to_csv('train_num.csv')
        except Exception as e:
            msg,typ,lin=sys.exc_info()
            logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')

    def feature_selection(self):
        try:
            self.x_train_num_col,self.x_test_num_col=fm(self.x_train_num_col,self.x_test_num_col,self.y_train,self.y_test)
        except Exception as e:
            msg,typ,lin=sys.exc_info()
            logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')

    def cat_to_num(self):
        try:
            self.x_train_cat_col,self.x_test_cat_col=c_t_n(self.x_train_cat_col,self.x_test_cat_col)
            # combine data
            self.x_train_num_col.reset_index(drop=True, inplace=True)
            self.x_train_cat_col.reset_index(drop=True, inplace=True)
            self.x_test_num_col.reset_index(drop=True, inplace=True)
            self.x_test_cat_col.reset_index(drop=True, inplace=True)

            self.training_data=pd.concat([self.x_train_num_col,self.x_train_cat_col],axis=1)
            self.testing_data=pd.concat([self.x_test_num_col,self.x_test_cat_col],axis=1)

            logger.info(f'========================================================================================')

            logger.info((f'final training data : {self.training_data.shape}'))
            logger.info((f'{self.training_data.columns}'))
            logger.info(f'training data null values : {self.training_data.isnull().sum()}')

            logger.info(f'final testing data : {self.testing_data.shape}')
            logger.info((f'{self.testing_data.columns}'))
            logger.info(f'testing data null values : {self.testing_data.isnull().sum()}')
        except Exception as e:
            msg,typ,lin=sys.exc_info()
            logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')

    def data_balancing(self):
        try:
            logger.info(f'Number of Rows for GOOD customer {1} : {sum(self.y_train==1)}')
            logger.info(f'Number of Rows for BAD customer {0} : {sum(self.y_train==0)}')
            logger.info(f'Training data size : {self.training_data.shape}')

            sm = SMOTE(random_state=42)

            self.training_data_bal, self.y_train_bal = sm.fit_resample(self.training_data, self.y_train)

            logger.info(f'Number of Rows for GOOD customer {1} : {sum(self.y_train_bal == 1)}')
            logger.info(f'Number of Rows for BAD customer {0} : {sum(self.y_train_bal == 0)}')
            logger.info(f'Training data size : {self.training_data_bal.shape}')

            fs(self.training_data_bal, self.y_train_bal, self.testing_data, self.y_test)

        except Exception as e:
            msg,typ,lin=sys.exc_info()
            logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')

if __name__ == '__main__':
    try:
        obj=CHURN('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        obj.missing_values()
        obj.data_seperation()
        obj.variable_transformation()
        obj.feature_selection()
        obj.cat_to_num()
        obj.data_balancing()
    except Exception as e:
        msg, typ, lin = sys.exc_info()
        logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')