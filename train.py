from utils import *
from extract_features import *
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc

from pycox.models import CoxPH, CoxTime
from pycox.evaluation import EvalSurv
from sklearn_pandas import DataFrameMapper

import torch
from torch import nn
import torchtuples as tt

import argparse
import pandas as pd

class LSTMCox(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, n_layers, output_size, drop_prob=0.7):
    super().__init__()
    self.n_layers = n_layers
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=0, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_size, bias=False)

  def forward(self, input):
    lstm_out, _ = self.lstm(input)
    out = self.fc(lstm_out[:, -1, :])

    return out

def train_coxph(data_df, r_splits):
  c_index_at = []
  c_index_30 = []

  time_auc_30 = []
  time_auc_60 = []
  time_auc_365 = []

  for i in range(len(r_splits)):
    print("\nIteration %s"%(i))
    #DATA PREP
    df_train, df_val, df_test, df_test_30 = prepare_datasets(data_df, r_splits[i][2], r_splits[i][1], r_splits[i][0])

    (data_x, data_y), (val_x, val_y), (test_x, test_y), (test_30_x, test_30_y) = df2array(data_df, df_train, df_val, df_test, df_test_30)

    estimator = CoxPHSurvivalAnalysis(alpha=1e-04)
    estimator.fit(data_x, data_y)

    c_index_at.append(estimator.score(test_x, test_y))
    c_index_30.append(estimator.score(test_30_x, test_30_y))

    for time_x in [30, 60, 365]:
      t_auc, t_mean_auc = cumulative_dynamic_auc(data_y, test_y, estimator.predict(test_x), time_x)
      eval("time_auc_" + str(time_x)).append(t_auc[0])

    print("C-index_30:", c_index_30[i])
    print("C-index_AT:", c_index_at[i])

    print("time_auc_30", time_auc_30[i])
    print("time_auc_60", time_auc_60[i])
    print("time_auc_365", time_auc_365[i]) 

  return c_index_at, c_index_30, time_auc_30, time_auc_60, time_auc_365

def train_deepsurv(data_df, r_splits):
  epochs = 100
  verbose = True

  num_nodes = [32]
  out_features = 1
  batch_norm = True
  dropout = 0.6
  output_bias = False

  c_index_at = []
  c_index_30 = []

  time_auc_30 = []
  time_auc_60 = []
  time_auc_365 = []

  for i in range(len(r_splits)):
    print("\nIteration %s"%(i))
    
    #DATA PREP
    df_train, df_val, df_test, df_test_30 = prepare_datasets(data_df, r_splits[i][2], r_splits[i][1], r_splits[i][0])
    
    xcols = list(df_train.columns)

    for col_name in ["subject_id", "event", "duration"]:
      if col_name in xcols:
        xcols.remove(col_name)

    cols_standardize = xcols

    standardize = [([col], StandardScaler()) for col in cols_standardize]

    x_mapper = DataFrameMapper(standardize)

    x_train = x_mapper.fit_transform(df_train).astype('float32')
    x_val = x_mapper.transform(df_val).astype('float32')
    x_test = x_mapper.transform(df_test).astype('float32')
    x_test_30 =  x_mapper.transform(df_test_30).astype('float32')

    labtrans = CoxTime.label_transform()
    get_target = lambda df: (df['duration'].values, df['event'].values)
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_val = labtrans.transform(*get_target(df_val))

    durations_test, events_test = get_target(df_test)
    durations_test_30, events_test_30 = get_target(df_test_30)
    val = tt.tuplefy(x_val, y_val)

    (train_x, train_y), (val_x, val_y), (test_x, test_y), _ = df2array(data_df, df_train, df_val, df_test, df_test_30)

    #MODEL
    in_features = x_train.shape[1]

    callbacks = [tt.callbacks.EarlyStopping()]

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout, output_bias=output_bias)

    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(0.0001)

    if x_train.shape[0] % 2:
      batch_size = 255
    else:
      batch_size = 256

    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val, val_batch_size=batch_size)

    model.compute_baseline_hazards()

    surv = model.predict_surv_df(x_test)
    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    c_index_at.append(ev.concordance_td())

    surv_30 = model.predict_surv_df(x_test_30)
    ev_30 = EvalSurv(surv_30, durations_test_30, events_test_30, censor_surv='km')
    c_index_30.append(ev_30.concordance_td())

    for time_x in [30, 60, 365]:
      va_auc, va_mean_auc = cumulative_dynamic_auc(train_y, test_y, model.predict(x_test).flatten(), time_x)

      eval("time_auc_" + str(time_x)).append(va_auc[0])

    print("C-index_30:", c_index_30[i])
    print("C-index_AT:", c_index_at[i])

    print("time_auc_30", time_auc_30[i])
    print("time_auc_60", time_auc_60[i])
    print("time_auc_365", time_auc_365[i])

  return c_index_at, c_index_30, time_auc_30, time_auc_60, time_auc_365

def train_LSTMCox(data_df, r_splits):
  epochs = 100
  verbose = True

  in_features = 768
  out_features = 1
  batch_norm = True
  dropout = 0.6
  output_bias = False

  c_index_at = []
  c_index_30 = []

  time_auc_30 = []
  time_auc_60 = []
  time_auc_365 = []

  for i in range(len(r_splits)):
    print("\nIteration %s"%(i))
    
    #DATA PREP
    df_train, df_val, df_test, df_test_30 = prepare_datasets(data_df, r_splits[i][2], r_splits[i][1], r_splits[i][0])

    x_train = np.array(df_train["x0"].tolist()).astype("float32")
    x_val = np.array(df_val["x0"].tolist()).astype("float32")
    x_test = np.array(df_test["x0"].tolist()).astype("float32")
    x_test_30 = np.array(df_test_30["x0"].tolist()).astype("float32")

    labtrans = CoxTime.label_transform()
    get_target = lambda df: (df['duration'].values, df['event'].values)
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_val = labtrans.transform(*get_target(df_val))

    durations_test, events_test = get_target(df_test)
    durations_test_30, events_test_30 = get_target(df_test_30)
    val = tt.tuplefy(x_val, y_val)
    
    (train_x, train_y), (val_x, val_y), (test_x, test_y), _ = df2array(data_df, df_train, df_val, df_test, df_test_30)

    #MODEL
    callbacks = [tt.callbacks.EarlyStopping()]

    net = LSTMCox(768, 32, 1, 1)

    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(0.0001)

    if x_train.shape[0] % 2:
      batch_size = 255
    else:
      batch_size = 256
      
    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val, val_batch_size=batch_size)

    model.compute_baseline_hazards()

    surv = model.predict_surv_df(x_test)
    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    c_index_at.append(ev.concordance_td())

    surv_30 = model.predict_surv_df(x_test_30)
    ev_30 = EvalSurv(surv_30, durations_test_30, events_test_30, censor_surv='km')
    c_index_30.append(ev_30.concordance_td())

    for time_x in [30, 60, 365]:
      va_auc, va_mean_auc = cumulative_dynamic_auc(train_y, test_y, model.predict(x_test).flatten(), time_x)

      eval("time_auc_" + str(time_x)).append(va_auc[0])

    print("C-index_30:", c_index_30[i])
    print("C-index_AT:", c_index_at[i])

    print("time_auc_30", time_auc_30[i])
    print("time_auc_60", time_auc_60[i])
    print("time_auc_365", time_auc_365[i])

  return c_index_at, c_index_30, time_auc_30, time_auc_60, time_auc_365

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--timeline', type=str, required=True)
  parser.add_argument('--feature_set', type=str, required=True)
  parser.add_argument('--feature_path', type=str, required=True)
  parser.add_argument('--model', type=str, required=True)
  parser.add_argument('--mapping', type=str, required=True)

  args = parser.parse_args()

  return args

if __name__ == '__main__':
  args = get_args()
  timelines = return_timelines(args.timeline)
  mapping = return_mapping(args.mapping)
  cv_splits = get_splits()

  print("Preparing Dataset")
  if args.feature_set in "label":
    data_df = format_labels(args.feature_path, timelines, mapping)
  elif args.feature_set == "hidden":
    data_df = format_hidden_features(args.feature_path, timelines, mapping)
  elif args.feature_set == "hidden_sequence":
    data_df = format_hf_sequence(args.feature_path, timelines, mapping)

  if args.model == "coxph":
    c_index_at, c_index_30, time_auc_30, time_auc_60, time_auc_365 = train_coxph(data_df, cv_splits)
  elif args.model == "deepsurv":
    c_index_at, c_index_30, time_auc_30, time_auc_60, time_auc_365 = train_deepsurv(data_df, cv_splits)
  elif args.model == "lstm_cox":
    c_index_at, c_index_30, time_auc_30, time_auc_60, time_auc_365 = train_LSTMCox(data_df, cv_splits)

  for data in ["c_index_30", "c_index_at", "time_auc_30", "time_auc_60", "time_auc_365"]:
    center, pm = mean_confidence_interval(eval(data))
    print("\n{}: {}".format(data, str("%.3f"%(center)) + " ± " + str("%.3f"%(pm))))
