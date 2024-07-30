# Retrain Lasso models on tuned hyperparameters:
# Lasso(alpha = 0.01, max_iter = 1000000, random_state = 0, tol = 0.01)
# And then predict
# Author: Max Korbmacher (max.korbmacher@gmail.com)
# 30 July 2024
#
###### ADD PATH FOLDER WHERE FILES WILL BE SAVED
savepath="/Users/max/Library/CloudStorage/OneDrive-HøgskulenpåVestlandet/Documents/Projects/LongBAG/results/prediction/"
# define data paths
datapath = "/Users/max/Library/CloudStorage/OneDrive-HøgskulenpåVestlandet/Documents/Projects/LongBAG/data/unscaled/"
#
###### DEFINE THE NUMBER OF REPETITIONS OF THE RANDOM SAMPLING PROCEDURE
###########################
#
# import packages
#
import csv
import pandas as pd
from functools import reduce
import numpy as np
#import pingouin as pg
#from pingouin import partial_corr
#from pingouin import logistic_regression
import scipy
from scipy.stats.stats import pearsonr
import time
from numpy import mean
from numpy import std
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, roc_auc_score, recall_score, precision_score, f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
from sklearn.feature_selection import r_regression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.utils import shuffle
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.lines as mlines
from scipy import stats
import sys, os
import statsmodels.api as sm
import json
from sklearn.linear_model import Lasso
import random
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
import copy
#
#Using this to get LaTeX font for plots (LaTeX code rules must be followed for e.g. axis titles (no underscores etc without \))
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)
from timeit import default_timer as timer
from datetime import timedelta
print("Timer starts now.")
start = timer()
#
####################################
print("Start loading data.")
# load data
T1 = pd.read_csv(datapath+'T1w_train.csv')
dMRI = pd.read_csv(datapath+'dMRI_train.csv')
multi = pd.read_csv(datapath+'multi_train.csv')
print("Training data loaded.")
# load test data
T1_test1 = pd.read_csv(datapath+'T1w_test1.csv')
T1_test2 = pd.read_csv(datapath+'T1w_test2.csv')
dMRI_test1 = pd.read_csv(datapath+'dMRI_test1.csv')
dMRI_test2 = pd.read_csv(datapath+'dMRI_test2.csv')
multi_test1 = pd.read_csv(datapath+'multi_test1.csv')
multi_test2 = pd.read_csv(datapath+'multi_test2.csv')
print("Test data loaded.")
#
# sort frames
T1 = T1.reindex(sorted(T1.columns), axis = 1)
dMRI = dMRI.reindex(sorted(dMRI.columns), axis = 1)
multi = multi.reindex(sorted(multi.columns), axis = 1)
T1_test1 = T1_test1.reindex(sorted(T1_test1.columns), axis = 1)
T1_test2 = T1_test2.reindex(sorted(T1_test2.columns), axis = 1)
dMRI_test1 = dMRI_test1.reindex(sorted(dMRI_test1.columns), axis = 1)
dMRI_test2 = dMRI_test2.reindex(sorted(dMRI_test2.columns), axis = 1)
multi_test1 = multi_test1.reindex(sorted(multi_test1.columns), axis = 1)
multi_test2 = multi_test2.reindex(sorted(multi_test2.columns), axis = 1)
#
T1_test1 = T1_test1.sort_values(by=['eid'])
T1_test2 = T1_test2.sort_values(by=['eid'])
dMRI_test1 = dMRI_test1.sort_values(by=['eid'])
dMRI_test2 = dMRI_test2.sort_values(by=['eid'])
multi_test1 = multi_test1.sort_values(by=['eid'])
multi_test2 = multi_test2.sort_values(by=['eid'])

eid_test1 = copy.deepcopy(multi_test1['eid'])
eid_test2 = copy.deepcopy(multi_test2['eid'])
Age_test1 = copy.deepcopy(multi_test1['age'])
Age_test2 = copy.deepcopy(multi_test2['age'])
# Then remove demographics from test sets for later prediction
#
T1_test1_1 = T1_test1.drop('eid',axis = 1)
T1_test1_1 = T1_test1_1.drop('age',axis = 1)
T1_test1_1 = T1_test1_1.drop('sex',axis = 1)
T1_test1_1 = T1_test1_1.drop('site',axis = 1)
T1_test2_1 = T1_test2.drop('eid',axis = 1)
T1_test2_1 = T1_test2_1.drop('age',axis = 1)
T1_test2_1 = T1_test2_1.drop('sex',axis = 1)
T1_test2_1 = T1_test2_1.drop('site',axis = 1)
#
dMRI_test1_1 = dMRI_test1.drop('eid',axis = 1)
dMRI_test1_1 = dMRI_test1_1.drop('age',axis = 1)
dMRI_test1_1 = dMRI_test1_1.drop('sex',axis = 1)
dMRI_test1_1 = dMRI_test1_1.drop('site',axis = 1)
dMRI_test2_1 = dMRI_test2.drop('eid',axis = 1)
dMRI_test2_1 = dMRI_test2_1.drop('age',axis = 1)
dMRI_test2_1 = dMRI_test2_1.drop('sex',axis = 1)
dMRI_test2_1 = dMRI_test2_1.drop('site',axis = 1)
#
multi_test1_1 = multi_test1.drop('eid',axis = 1)
multi_test1_1 = multi_test1_1.drop('age',axis = 1)
multi_test1_1 = multi_test1_1.drop('sex',axis = 1)
multi_test1_1 = multi_test1_1.drop('site',axis = 1)
multi_test2_1 = multi_test2.drop('eid',axis = 1)
multi_test2_1 = multi_test2_1.drop('age',axis = 1)
multi_test2_1 = multi_test2_1.drop('sex',axis = 1)
multi_test2_1 = multi_test2_1.drop('site',axis = 1)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# for testing of the code, I use fractions of the data
#T1 = T1.sample(frac=0.001)
#dMRI = dMRI.sample(frac=0.001)
#multi = multi.sample(frac=0.001)
#
#
#
#
T1_label = T1['age']
T1_train = T1.copy()
T1_train = T1_train.drop('eid',axis = 1)
T1_train = T1_train.drop('age',axis = 1)
T1_train = T1_train.drop('sex',axis = 1)
T1_train = T1_train.drop('site',axis = 1)
#
dMRI_label = dMRI['age']
dMRI_train = dMRI.copy()
dMRI_train = dMRI_train.drop('eid',axis = 1)
dMRI_train = dMRI_train.drop('age',axis = 1)
dMRI_train = dMRI_train.drop('sex',axis = 1)
dMRI_train = dMRI_train.drop('site',axis = 1)
#
multi_label = multi['age']
multi_train = multi.copy()
multi_train = multi_train.drop('eid',axis = 1)
multi_train = multi_train.drop('age',axis = 1)
multi_train = multi_train.drop('sex',axis = 1)
multi_train = multi_train.drop('site',axis = 1)
#
# define models, based on hyperparamter tuning
our_model = Lasso(alpha = 0.01, max_iter = 1000000, random_state = 0, tol = 0.01)
our_model2 = Lasso(alpha = 0.01, max_iter = 1000000, random_state = 0, tol = 0.01)
our_model3 = Lasso(alpha = 0.01, max_iter = 1000000, random_state = 0, tol = 0.01)
#
print("Start training.")
#
# fit the model for predictions in test data
T1_model = our_model.fit(T1_train,T1_label)
dMRI_model = our_model2.fit(dMRI_train, dMRI_label)
multi_model = our_model3.fit(multi_train, multi_label)
# Save the model on the disk (if this is wished)
with open(savepath+ 'Lasso_T1_model_%s.pkl','wb') as f:
    pickle.dump(T1_model,f)
with open(savepath+ 'Lasso_dMRI_model_%s.pkl','wb') as f:
    pickle.dump(dMRI_model,f)
with open(savepath+ 'Lasso_multi_model_%s.pkl','wb') as f:
    pickle.dump(multi_model,f)
# make predictions
print("Training completed. Predictions are now made.")  
# predictions in training data (deemed not useful here, due to multiple random samping. Instead, we use a validation set.)
T1['T1_pred'] = T1_model.predict(T1_train)
T1_test1['T1_pred'] = T1_model.predict(T1_test1_1)
T1_test2['T1_pred'] = T1_model.predict(T1_test2_1)

T1['dMRI_pred'] = dMRI_model.predict(dMRI_train)
T1_test1['dMRI_pred'] = dMRI_model.predict(dMRI_test1_1)
T1_test2['dMRI_pred'] = dMRI_model.predict(dMRI_test2_1)

T1['multi_pred'] = multi_model.predict(multi_train)
T1_test1['multi_pred'] = multi_model.predict(multi_test1_1)
T1_test2['multi_pred'] = multi_model.predict(multi_test2_1)

train_save = T1[['eid', 'age', 'sex', 'site', 'T1_pred', 'dMRI_pred', 'multi_pred']]
test1_save = T1_test1[['eid', 'age', 'sex', 'site', 'T1_pred', 'dMRI_pred', 'multi_pred']]
test2_save = T1_test2[['eid', 'age', 'sex', 'site', 'T1_pred', 'dMRI_pred', 'multi_pred']]

train_save.to_csv(savepath+'Lasso_training_predictions.csv')
test1_save.to_csv(savepath+'Lasso_test1_predictions.csv')
test2_save.to_csv(savepath+'Lasso_test2_predictions.csv')

print("Done all training, testing, and saving the outputs.")

print("This is the end.")
print("Elapsed time:")
end = timer()
print(timedelta(seconds=end-start))
print("########################")
print("Interpretation time!")
