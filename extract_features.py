import pandas as pd
import numpy as np
from utils import *
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import re

def format_labels(file_path, timelines, mapping):
	most_recent = mapping.sort_values(["subject_id", "ordering_date"], ascending=False).drop_duplicates("subject_id", keep="first")

	label_features = pd.read_csv(file_path)
	formatted_features = reformat4pycox(["report_id"], label_features)

	#Connect subject to report
	data_frames = [timelines, most_recent]
	data_df = reduce(lambda  left,right: pd.merge(left,right,on="subject_id"), data_frames)

	#Connect report to labels
	data_frames = [data_df, formatted_features]
	data_df = reduce(lambda  left,right: pd.merge(left,right,on="report_id"), data_frames)

	for i in ["ordering_date", "report_id"]:
	  del data_df[i]

	return data_df

def format_hidden_features(file_path, timelines, mapping):
	loaded = np.load(file_path)

	most_recent = mapping.sort_values(["subject_id", "ordering_date"], ascending=False).drop_duplicates("subject_id", keep="first")
	report_ids = list(most_recent['report_id'])

	mutable_file = {}  
	for id in report_ids:
	  mutable_file[id] = loaded[id].flatten()
	loaded = mutable_file

	label_features = pd.DataFrame(loaded.values(), index=loaded)

	cols = list(label_features.columns)
	xcols = ["x" + str(i) for i in cols]
	rename_dict = dict(zip(cols,xcols))
	rename_dict["index"] = "report_id"

	label_features = label_features.reset_index().rename(columns=rename_dict)

	#Connect subject to report
	data_frames = [timelines, most_recent]
	data_df = reduce(lambda  left,right: pd.merge(left,right,on="subject_id"), data_frames)

	#Connect report to labels
	data_frames = [data_df, label_features]
	data_df = reduce(lambda  left,right: pd.merge(left,right,on="report_id"), data_frames)

	for i in ["ordering_date", "report_id"]:
	  del data_df[i]

	return data_df

def format_hf_sequence(file_path, timelines, mapping):
	loaded = np.load(file_path)
	 
	top3_reports = mapping.sort_values(["subject_id", "ordering_date"], ascending=True).groupby("subject_id").tail(3)

	#Create a list of report ids
	report_dict = top3_reports.groupby("subject_id")["report_id"].apply(list).to_dict()

	#Create a dict of report arrays. Format: key: array of report embeddings
	embedding_dict = defaultdict(list)

	for k,v in report_dict.items():
		for vi in v:
		  embedding_dict[k].append(loaded[vi])

		embedding_dict[k] = np.vstack(embedding_dict[k])

	#Converting embedding dict into dataframe
	label_features = pd.DataFrame(embedding_dict.values(), index=embedding_dict)

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
	label_features = label_features.reset_index().rename(columns={"index": "subject_id"})

	data_frames = [timelines, label_features]
	data_df = reduce(lambda  left,right: pd.merge(left,right,on="subject_id"), data_frames)

	return data_df
