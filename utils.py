from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import re
import itertools
from functools import reduce
from collections import Counter

def reformat4pycox(non_cv, dataframe):
  cols = list(dataframe.columns)
  
  counter = 0 
  rename_dict = {}

  for i in cols:
    if i not in non_cv:
      rename_dict[i] = "x" + str(counter)
      counter+=1

  dataframe = dataframe.rename(columns=rename_dict)

  return dataframe

def mean_confidence_interval(data):
  errors = np.array(data)
  m = errors.mean()
  r = 1.960 * (np.std(errors, ddof=1) / np.sqrt(len(errors)))

  return m, r

def prepare_datasets(data_df, training_indices, val_indices, test_indices):
  df_train = data_df[data_df["fold_index"].isin(training_indices)].drop(['fold_index'], axis = 1).copy(deep=True) 
  df_val = data_df[data_df["fold_index"].isin(val_indices)].drop(['fold_index'], axis = 1).copy(deep=True) 
  df_test = data_df[data_df["fold_index"].isin(test_indices)].drop(['fold_index'], axis = 1).copy(deep=True) 
  df_test_30 = data_df[data_df["fold_index"].isin(test_indices)].drop(['fold_index'], axis = 1).copy(deep=True) 
  df_test_30.loc[(df_test_30.duration > 30),'event'] = False

  return df_train, df_val, df_test, df_test_30

def df2array(data_df, df_train, df_val, df_test, df_test_30):
  cols = list(data_df.columns)
  r = re.compile("x\d+")
  covariates = list(filter(r.match, cols)) 

  train_x = df_train[covariates]

  train_y = df_train[["event", "duration"]].to_numpy()
  train_y = np.array([tuple(x) for x in train_y.tolist()], dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

  val_x = df_val[covariates]

  val_y = df_val[["event", "duration"]].to_numpy()
  val_y = np.array([tuple(x) for x in val_y.tolist()], dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

  test_x = df_test[covariates]

  test_y = df_test[["event", "duration"]].to_numpy()
  test_y = np.array([tuple(x) for x in test_y.tolist()], dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

  test_30_x = df_test_30[covariates]

  test_30_y = df_test_30[["event", "duration"]].to_numpy()
  test_30_y = np.array([tuple(x) for x in test_30_y.tolist()], dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

  return (train_x, train_y), (val_x, val_y), (test_x, test_y), (test_30_x, test_30_y)

def add_paddings(arrs):
  arrs = arrs.tolist()

  if len(arrs) < 3:
    for i in range(3 - len(arrs)):
      arrs.insert(0, [0] *len(arrs[0]))

  return np.array(arrs)

def normalize_data(data_df):
  data_df["x0"] = data_df["x0"].apply(lambda x: add_paddings(x))

  list2d = data_df["x0"]

  merged = list(itertools.chain(*list2d))

  scaler = StandardScaler()
  scaler.fit(merged)

  data_df["x0"] = data_df["x0"].apply(lambda x: scaler.transform(x))

  return data_df

def return_mapping(file_path):
  mapping = pd.read_csv(file_path)
  mapping['subject_id'] = mapping['subject_id'].apply(lambda x: x.strip("s"))

  return mapping

def return_timelines(file_path):
  timelines = pd.read_csv(file_path)
  timelines = timelines.rename(columns={"time_to_event": "duration"})
  timelines = timelines[["subject_id", "event", "duration", "fold_index"]]
  timelines = timelines[~(timelines["duration"] < 0)].dropna()
  timelines['subject_id'] = timelines['subject_id'].apply(lambda x: x.strip("s"))

  return timelines

def get_splits():
  r_splits = []

  for i in range(0,10,2):
    test_set = [i, i+1]
    val_set = [(i-1)%10]
    train_set = [a for a in list(range(10)) if a not in test_set and a not in val_set]
    r_splits.append((test_set, val_set, train_set))

  return r_splits