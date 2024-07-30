# load CV XGB and Lasso models and predict
# (linear models are handled in R in other code snippet)
# (Lasso models are handled in Python in another code snippet)
# Max Korbmacher, 30th July 2024
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import r2_score
import pickle
import os
import numpy as np
import scipy.sparse as sparse
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb
#
# Define paths
## for the data to be predicted on
datapath = "/Users/max/Library/CloudStorage/OneDrive-HøgskulenpåVestlandet/Documents/Projects/LongBAG/data/unscaled/"
## where the models are
modelpath = "/Users/max/Library/CloudStorage/OneDrive-HøgskulenpåVestlandet/Documents/Projects/LongBAG/results/unscaled/CV_models/"
#
## where the results are supposed to be saved
savepath = "/Users/max/Library/CloudStorage/OneDrive-HøgskulenpåVestlandet/Documents/Projects/LongBAG/results/prediction/"

# load training
T1 = pd.read_csv(datapath+'T1w_train.csv')
dMRI = pd.read_csv(datapath+'dMRI_train.csv')
multi = pd.read_csv(datapath+'multi_train.csv')

# load test data
T1_test1 = pd.read_csv(datapath+'T1w_test1.csv')
T1_test2 = pd.read_csv(datapath+'T1w_test2.csv')
#
dMRI_test1 = pd.read_csv(datapath+'dMRI_test1.csv')
dMRI_test2 = pd.read_csv(datapath+'dMRI_test2.csv')
#
multi_test1 = pd.read_csv(datapath+'multi_test1.csv')
multi_test2 = pd.read_csv(datapath+'multi_test2.csv')
#
# sort data
T1_test1 = T1_test1.sort_values(by=['eid'])
T1_test2 = T1_test2.sort_values(by=['eid'])
dMRI_test1 = dMRI_test1.sort_values(by=['eid'])
dMRI_test2 = dMRI_test2.sort_values(by=['eid'])
multi_test1 = multi_test1.sort_values(by=['eid'])
multi_test2 = multi_test2.sort_values(by=['eid'])
#
# remove demographics for prediction
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
# add simple counter
T1_train['X'] = range(len(T1_train))
T1_test1_1['X'] = range(len(T1_test1_1))
T1_test2_1['X'] = range(len(T1_test2_1))
dMRI_train['X'] = range(len(dMRI_train))
dMRI_test1_1['X'] = range(len(dMRI_test1_1))
dMRI_test2_1['X'] = range(len(dMRI_test2_1))
multi_train['X'] = range(len(multi_train))
multi_test1_1['X'] = range(len(multi_test1_1))
multi_test2_1['X'] = range(len(multi_test2_1))

# load xgb models
T1_model = xgb.XGBRegressor()
dMRI_model = xgb.XGBRegressor()
multi_model = xgb.XGBRegressor()
T1_model.load_model(modelpath + "XGB_T1w_train_model.txt")
#dMRI_model.load_model(modelpath + "XGB_dMRI_train_model.txt")
multi_model.load_model(modelpath + "XGB_multi_train_model.txt")
#
print("Data and models are loaded. Now refitting models on training data.")
# fit
T1_model = T1_model.fit(T1_train, T1_label)
print("Model fitted on T1w data.")
dMRI_model = dMRI_model.fit(dMRI_train, dMRI_label)
print("Model fitted on dMRI data.")
multi_model = multi_model.fit(multi_train, multi_label)
print("Model fitted on multimodal data.")
print("Fitting completed. Predictions start.")
#
# predict
T1['T1_pred'] = T1_model.predict(T1_train)
T1_test1['T1_pred'] = T1_model.predict(T1_test1_1)
T1_test2['T1_pred'] = T1_model.predict(T1_test2_1)
#
T1['dMRI_pred'] = dMRI_model.predict(dMRI_train)
T1_test1['dMRI_pred'] = dMRI_model.predict(dMRI_test1_1)
T1_test2['dMRI_pred'] = dMRI_model.predict(dMRI_test2_1)
#
T1['multi_pred'] = multi_model.predict(multi_train)
T1_test1['multi_pred'] = multi_model.predict(multi_test1_1)
T1_test2['multi_pred'] = multi_model.predict(multi_test2_1)

train_save = T1_train[['eid', 'age', 'sex', 'site', 'T1_pred', 'dMRI_pred', 'multi_pred']]
test1_save = T1_test1[['eid', 'age', 'sex', 'site', 'T1_pred', 'dMRI_pred', 'multi_pred']]
test2_save = T1_test2[['eid', 'age', 'sex', 'site', 'T1_pred', 'dMRI_pred', 'multi_pred']]

train_save.to_csv(savepath+'XGB_training_predictions.csv')
test1_save.to_csv(savepath+'XGB_test1_predictions.csv')
test2_save.to_csv(savepath+'XGB_test2_predictions.csv')
#
#
print("######################")
print("All done.")
print("######################")



