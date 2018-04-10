import pandas as pd
# import ray.dataframe as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc

def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'dart',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.01,
        'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 0,
        'metric':metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=50, 
                     feval=feval)

    n_estimators = bst1.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    return bst1

path = '/media/alaa/Study/TalkingData_FraudDetection_challenge/data/talkingdata-adtracking-fraud-detection/'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

print('load train...')
train_df = pd.read_csv(path+"train.csv", nrows=55000000, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed']) #, nrows=35000000
train_df = train_df.reindex(
    np.random.permutation(train_df.index)) # randomize the order of the training data
print('load test...')
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])


len_train = len(train_df)
train_df=train_df.append(test_df)

del test_df
gc.collect()

print('data prep...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')

gc.collect()

print('group by...')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})

print('merge...')
train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')

# print train_df.columns

def linear_scale(series):
  min_val = series.min()
  max_val = series.max()
  scale = (max_val - min_val) / 2.0
  return series.apply(lambda x:((x - min_val) / scale) - 1.0)

def normalize_linear_scale(examples_dataframe):
  """Returns a version of the input `DataFrame` that has 
     all its features normalized linearly."""
  return examples_dataframe.apply(linear_scale, axis=0)

def log_normalize(series):
  return series.apply(lambda x:math.log(abs(x+1.0)))

def clip(series, clip_to_min, clip_to_max):
  return series.apply(lambda x:(
    min(max(x, clip_to_min), clip_to_max)))

def z_score_normalize(series):
  mean = series.mean()
  std_dv = series.std()
  return series.apply(lambda x:(x - mean) / std_dv)

def binary_threshold(series, threshold):
  return series.apply(lambda x:(1 if x > threshold else 0))

def normalize(examples_dataframe, norm_type, clip_to_min=None, clip_to_max=None, threshold=None):
  """
  norm_type = 'log', 'clipping', 'z_score_norm', or 'binary_threshold'
    clip_to_min & clip_to_max only used in case of norm_type='clipping'
    threshold only used in case of norm_type='binary_threshold'
    
  Note: this function could be used to do different kind of normalization on different features
  by calling it multiple time for every group of features while specifying norm_type
  Returns a version of the input `DataFrame` that has all its features normalized."""
  if norm_type == 'log':
    return examples_dataframe.apply(log_normalize, axis=0)
  elif norm_type == 'clipping':
    return examples_dataframe.apply(clip, args=(clip_to_min, clip_to_max), axis=0)
  elif norm_type == 'z_score_norm':
    return examples_dataframe.apply(z_score_normalize, axis=0)
  elif norm_type == 'binary_threshold':
    return examples_dataframe.apply(binary_threshold, args=(threshold), axis=0)

# normalized_dataframe = normalize(preprocess_features(training_examples), norm_type='log')
# normalized_dataframe = normalize(preprocess_features(california_housing_dataframe), clip_to_min=0, clip_to_max=1, norm_type='clipping')
# normalized_dataframe = normalize(preprocess_features(training_examples), norm_type='z_score_norm')

print('data normalization ..')
# train_df_norm_1 = normalize(train_df['ip'], norm_type='z_score_norm') 
# train_df_norm_2 = normalize(train_df['device'], norm_type='z_score_norm') 

# train_df = pd.concat([train_df[u'app', u'channel', u'click_id', u'click_time',u'is_attributed', u'os', u'hour', u'day', u'qty'], train_df_norm_1, train_df_norm_2])
# train_df_norm_1 = normalize(train_df, norm_type='z_score_norm') 

train_df.info()

test_df = train_df[len_train:]
print(len(test_df))
val_df = train_df[(len_train-3000000):len_train]
print(len(val_df))
train_df = train_df[:(len_train-3000000)]
print(len(train_df))

target = 'is_attributed'
predictors = ['app','device','os', 'channel', 'hour', 'qty']
categorical = ['app','device','os', 'channel', 'hour']


sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

gc.collect()

print("Training...")
params = {
    'learning_rate': 0.1,
    'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 1400,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 4,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': .7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    # 'scale_pos_weight':200, # because training data is extremely unbalanced 
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0, # L2 regularization term on weights
}
bst = lgb_modelfit_nocv(params, 
                        train_df, 
                        val_df, 
                        predictors, 
                        target, 
                        objective='binary', 
                        metrics='auc',
                        early_stopping_rounds=30, 
                        verbose_eval=True, 
                        num_boost_round=1000, 
                        categorical_features=categorical) # the returned model will be the best iteration model ever

del train_df
del val_df
gc.collect()

print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors])
filename = 'sub_lgb_balanced99_trail_11.csv'
print("writing in...", filename)
sub.to_csv(filename,index=False)
print("done...")
print(sub.info())




"""
--- 1st Trial
[163]   train's auc: 0.973114   valid's auc: 0.968485 # [163] means that the algorithm early stopped @ iteration #163 
--- 2nd Trial
[195]   train's auc: 0.973612   valid's auc: 0.968642
--- 3rd Trial
[195]   train's auc: 0.973533   valid's auc: 0.96859
--- 4th Trial >>> FAILED
--- 5th Trial 
[163] train's auc: 0.973114 valid's auc: 0.96848
--- 6th Trial
max_bin = 500 instead of 100
[82]  train's auc: 0.971377 valid's auc: 0.968496
--- 7th Trial
num_boost_round=1000, learning_rate=0.001, max_bin=100
--- 8th Trial
learning_rate=0.1, 'dart', num_leaves=10000
[292] train's auc: 0.971616 valid's auc: 0.968932
--- 9th Trial
num_leaves=10000, 'max_depth': 10, 'scale_pos_weight':200
"""

"""

    Use large max_bin (may be slower)
    Use small learning_rate with large num_iterations
    Use large num_leaves (may cause over-fitting)
    Use bigger training data
    Try dart

"""