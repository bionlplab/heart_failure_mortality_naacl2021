import pandas as pd
import numpy as np
from utils import *
from sklearn.preprocessing import StandardScaler
import re


def format_cnn_labels(file_path, timelines, splits):
	nlp_outputs = pd.read_csv(file_path)
	data_frames = [timelines, nlp_outputs, splits]
	data_df = reduce(lambda  left,right: pd.merge(left,right,on="mrn"), data_frames)

	data_df = reformat4pycox(["mrn", "event", "duration", "fold_index"], data_df)

	return data_df

def format_chexbert_labels(file_path, timelines, splits):
	label_features = pd.read_csv(file_path).rename(columns={"id": "mrn", "Report Impression": "report"})

	formatted_features = reformat4pycox(["mrn", "report"], label_features)

	data_frames = [timelines, formatted_features, splits]
	data_df = reduce(lambda  left,right: pd.merge(left,right,on="mrn"), data_frames)

	return data_df

def format_hidden_features(file_path, timelines, splits):
	if re.search("|".join(["chexbert_hidden.npz", "bluebert_hidden.npz", "generalbert_hidden.npz"]), file_path):
		loaded = np.load(file_path)
		mutable_file = dict(loaded)
		for k,v in mutable_file.items():
		  mutable_file[k] = v.flatten()
		loaded = mutable_file
	elif re.search("|".join(["biobert_hidden.npz", "clinicalbert_hidden.npz"]),file_path):
		loaded = np.load(file_path)
	else:
		raise ValueError("Incompatible File")

	label_features = pd.DataFrame(loaded.values(), index=loaded)

	cols = list(label_features.columns)
	xcols = ["x" + str(i) for i in cols]
	rename_dict = dict(zip(cols,xcols))
	rename_dict["index"] = "mrn"

	label_features = label_features.reset_index().rename(columns=rename_dict)
	label_features["mrn"] = label_features["mrn"].astype(int)

	formatted_features = reformat4pycox(["mrn", "report"], label_features)

	data_frames = [timelines, formatted_features, splits]
	data_df = reduce(lambda  left,right: pd.merge(left,right,on="mrn"), data_frames)

	return data_df

def format_hf_sequence(file_path, timelines, splits):
	loaded = np.load(file_path)

	label_features = pd.DataFrame(loaded.values(), index=loaded)
	label_features[0] = label_features[0].apply(lambda x: add_paddings(x))

	list2d = label_features[0]

	merged = list(itertools.chain(*list2d))

	scaler = StandardScaler()
	scaler.fit(merged)

	label_features[0] = label_features[0].apply(lambda x: scaler.transform(x))

	cols = list(label_features.columns)
	xcols = ["x" + str(i) for i in cols]
	rename_dict = dict(zip(cols,xcols))

	label_features = label_features.rename(columns=rename_dict)
	label_features = label_features.reset_index().rename(columns={"index": "mrn"})
	label_features["mrn"] = label_features["mrn"].astype(int)

	data_frames = [timelines, label_features, splits]
	data_df = reduce(lambda  left,right: pd.merge(left,right,on="mrn"), data_frames)

	return data_df
