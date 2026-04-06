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
logger=setup_logging('all_models')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve

from sklearn.model_selection import GridSearchCV,cross_validate

def knn(x_train,x_test,y_train,y_test):
  global knn_reg
  global knn_pred
  knn_reg=KNeighborsClassifier(n_neighbors=5)
  knn_reg.fit(x_train,y_train)
  knn_pred=knn_reg.predict(x_test)
  logger.info(confusion_matrix(y_test,knn_pred))
  logger.info(accuracy_score(y_test,knn_pred))
  logger.info(classification_report(y_test,knn_pred))
def nb(x_train,x_test,y_train,y_test):
  global nb_reg
  global nb_pred
  nb_reg=GaussianNB()
  nb_reg.fit(x_train,y_train)
  nb_pred=nb_reg.predict(x_test)
  logger.info(confusion_matrix(y_test,nb_pred))
  logger.info(accuracy_score(y_test,nb_pred))
  logger.info(classification_report(y_test,nb_pred))
def lr(x_train,x_test,y_train,y_test):
  global lr_reg
  global lr_pred
  lr_reg=LogisticRegression()
  lr_reg.fit(x_train,y_train)
  lr_pred=lr_reg.predict(x_test)
  logger.info(confusion_matrix(y_test,lr_pred))
  logger.info(accuracy_score(y_test,lr_pred))
  logger.info(classification_report(y_test,lr_pred))
def dt(x_train,x_test,y_train,y_test):
  global dt_reg
  global dt_pred
  dt_reg=DecisionTreeClassifier(criterion='entropy')
  dt_reg.fit(x_train,y_train)
  dt_pred=dt_reg.predict(x_test)
  logger.info(confusion_matrix(y_test,dt_pred))
  logger.info(accuracy_score(y_test,dt_pred))
  logger.info(classification_report(y_test,dt_pred))
def rf(x_train,x_test,y_train,y_test):
  global rf_reg
  global rf_pred
  rf_reg=RandomForestClassifier(criterion='entropy',n_estimators=5)
  rf_reg.fit(x_train,y_train)
  rf_pred=rf_reg.predict(x_test)
  logger.info(confusion_matrix(y_test,rf_pred))
  logger.info(accuracy_score(y_test,rf_pred))
  logger.info(classification_report(y_test,rf_pred))
def adab(x_train,x_test,y_train,y_test):
  global adab_reg
  global adab_pred
  lr=LogisticRegression()
  adab_reg=AdaBoostClassifier(estimator=lr,n_estimators=5)
  adab_reg.fit(x_train,y_train)
  adab_pred=adab_reg.predict(x_test)
  logger.info(confusion_matrix(y_test,adab_pred))
  logger.info(accuracy_score(y_test,adab_pred))
  logger.info(classification_report(y_test,adab_pred))
def gb(x_train,x_test,y_train,y_test):
  global gb_reg
  global gb_pred
  gb_reg=GradientBoostingClassifier(n_estimators=5)
  gb_reg.fit(x_train,y_train)
  gb_pred=gb_reg.predict(x_test)
  logger.info(confusion_matrix(y_test,gb_pred))
  logger.info(accuracy_score(y_test,gb_pred))
  logger.info(classification_report(y_test,gb_pred))
def xgb(x_train,x_test,y_train,y_test):
  global xgb_reg
  global xgb_pred
  xgb_reg=XGBClassifier(n_estimators=5)
  xgb_reg.fit(x_train,y_train)
  xgb_pred=xgb_reg.predict(x_test)
  logger.info(confusion_matrix(y_test,xgb_pred))
  logger.info(accuracy_score(y_test,xgb_pred))
  logger.info(classification_report(y_test,xgb_pred))
def svm(x_train,x_test,y_train,y_test):
  global svm_reg
  global svm_pred
  svm_reg=SVC(kernel='rbf')   # Radial Basis Function. (where separation becomes possible.) non linear to linear
  svm_reg.fit(x_train,y_train)
  svm_pred=svm_reg.predict(x_test)
  logger.info(confusion_matrix(y_test,svm_pred))
  logger.info(accuracy_score(y_test,svm_pred))
  logger.info(classification_report(y_test,svm_pred))
def auc_roc_tech(x_train,y_train,x_test,y_test):
    knn_fpr, knn_tpr, knn_th = roc_curve(y_test, knn_pred)
    nb_fpr, nb_tpr, nb_th = roc_curve(y_test, nb_pred)
    lr_fpr, lr_tpr, lr_th = roc_curve(y_test, lr_pred)
    dt_fpr, dt_tpr, dt_th = roc_curve(y_test, dt_pred)
    rf_fpr, rf_tpr, rf_th = roc_curve(y_test, rf_pred)
    adab_fpr, adab_tpr, adab_th = roc_curve(y_test, adab_pred)
    gb_fpr, gb_tpr, gb_th = roc_curve(y_test, gb_pred)
    xgb_fpr, xgb_tpr, xgb_th = roc_curve(y_test, xgb_pred)
    svm_fpr, svm_tpr, svm_th = roc_curve(y_test, svm_pred)

    plt.figure(figsize=(5, 3))
    plt.plot([0, 1], [0, 1])
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('all models auc curves')

    plt.plot(knn_fpr, knn_tpr, label='knn')
    plt.plot(nb_fpr, nb_tpr, label='nb')
    plt.plot(lr_fpr, lr_tpr, label='lr')
    plt.plot(dt_fpr, dt_tpr, label='dt')
    plt.plot(rf_fpr, rf_tpr, label='rf')
    plt.plot(adab_fpr, adab_tpr, label='adab')
    plt.plot(gb_fpr, gb_tpr, label='gb')
    plt.plot(xgb_fpr, xgb_tpr, label='xgb')
    plt.plot(svm_fpr, svm_tpr, label='svm')
    plt.legend(loc=0)
    plt.show()

def hypertuning(x_train,y_train,x_test,y_test):
    try:
        parameters_list = [
            # L2
            {
                'penalty': ['l2'],
                'solver': ['liblinear', 'lbfgs'],
                'C': [0.1, 1, 10],
                'max_iter': [100, 200],
                'class_weight': [None, 'balanced']
            },

            #  L1
            {
                'penalty': ['l1'],
                'solver': ['liblinear'],
                'C': [0.1, 1, 10],
                'max_iter': [100, 200],
                'class_weight': [None, 'balanced']
            }
        ]
        grid_reg = GridSearchCV(estimator=LogisticRegression(), param_grid=parameters_list, scoring='accuracy', cv=3)
        grid_result = grid_reg.fit(x_train, y_train)
        logger.info(f'The grid_result {grid_result}')
        logger.info(f'The grid best parameter are {grid_result.best_params_}')
        logger.info(f'The grid Accuracy Score {grid_result.best_score_}')

    except Exception as e:
        msg, typ, lin = sys.exc_info()
        logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')

def common(x_train,y_train,x_test,y_test):
    try:
        logger.info('------knn------')
        knn(x_train, x_test, y_train, y_test)
        logger.info('-----nb-----')
        nb(x_train, x_test, y_train, y_test)
        logger.info('------lr-----')
        lr(x_train, x_test, y_train, y_test)
        logger.info('----dt-----')
        dt(x_train, x_test, y_train, y_test)
        logger.info('-----rf------')
        rf(x_train, x_test, y_train, y_test)
        logger.info('------adab------')
        adab(x_train, x_test, y_train, y_test)
        logger.info('-------gb------')
        gb(x_train, x_test, y_train, y_test)
        logger.info('-----xgb------')
        xgb(x_train, x_test, y_train, y_test)
        logger.info('------svm-----')
        svm(x_train, x_test, y_train, y_test)
        logger.info(f'-----------auc_roc---------------')
        auc_roc_tech(x_train,y_train,x_test,y_test)
        logger.info(f'*********HYPERPARAMETER--TUNING*******************************')
        #hypertuning(x_train,y_train,x_test,y_test)

    except Exception as e:
        msg, typ, lin = sys.exc_info()
        logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')
