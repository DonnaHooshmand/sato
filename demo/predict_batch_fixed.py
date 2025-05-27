
### issue with this one: It has afixed max column value. otherwise it works perfectly fine.

import os
import torch
import argparse
import pandas as pd
import numpy as np
from os.path import join
from sklearn.preprocessing import LabelEncoder
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------
# Environment Configuration
# -------------------------
os.environ.setdefault('LDA_name', 'num-directstr_thr-0_tn-400')
os.environ.setdefault('TYPENAME', 'sato')
os.environ.setdefault('BASEPATH', './')

# -------------------------
# SATO Imports
# -------------------------
from extract.feature_extraction.topic_features_LDA import extract_topic_features
from extract.feature_extraction.sherlock_features import extract_sherlock_features
from utils import get_valid_types
from model import models_sherlock
from model.torchcrf import CRF

# -------------------------
# Constants
# -------------------------
MAX_COL_COUNT = 50
topic_dim = 400
device = 'cpu'

# -------------------------
# Type and Label Setup
# -------------------------
TYPENAME = os.environ['TYPENAME']
valid_types = get_valid_types(TYPENAME)
label_enc = LabelEncoder()
label_enc.fit(valid_types)

# -------------------------
# Feature Group Columns
# -------------------------
feature_group_cols = {}
sherlock_feature_groups = ['char', 'word', 'par', 'rest']
for f_g in sherlock_feature_groups:
    path = join(os.environ['BASEPATH'], 'configs', 'feature_groups', f"{f_g}_col.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature group file not found: {path}")
    feature_group_cols[f_g] = list(pd.read_csv(path, sep='\t', header=None, index_col=0)[1])

# -------------------------
# Utility: Pad Topic Vector
# -------------------------
pad_vec = lambda x: np.pad(x, (0, topic_dim - len(x)), 'constant', constant_values=(0.0, 1/topic_dim))

# -------------------------
# Load Pretrained Models
# -------------------------
pre_trained_loc = './pretrained_sato'
if not os.path.exists(join(pre_trained_loc, 'model.pt')):
    raise FileNotFoundError("Missing model.pt in pretrained_sato directory.")

classifier = models_sherlock.build_sherlock(sherlock_feature_groups, num_classes=len(valid_types),
                                            topic_dim=topic_dim, dropout_ratio=0.35)
model = CRF(len(valid_types), batch_first=True).to(device)

loaded_params = torch.load(join(pre_trained_loc, 'model.pt'), map_location=device)
classifier.load_state_dict(loaded_params['col_classifier'])
model.load_state_dict(loaded_params['CRF_model'])
classifier.eval()
model.eval()

# -------------------------
# Feature Extraction (now safer)
# -------------------------
def extract(df):
    df_dic = {'df': df, 'locator': 'None', 'dataset_id': 'None'}
    feature_dic = {}
    n = df.shape[1]

    try:
        topic_features = extract_topic_features(df_dic)
        if topic_features is None:
            raise ValueError("Topic feature extraction returned None.")
        topic_vec = pad_vec(topic_features.loc[0, 'table_topic'])
    except Exception as e:
        print(f"[ERROR] Topic feature extraction failed: {e}")
        topic_vec = np.zeros(topic_dim)

    feature_dic['topic'] = torch.FloatTensor(
        np.vstack((np.tile(topic_vec, (n, 1)), np.zeros((MAX_COL_COUNT - n, topic_dim))))
    )

    sherlock_features = extract_sherlock_features(df_dic)
    for f_g in feature_group_cols:
        temp = sherlock_features[feature_group_cols[f_g]].to_numpy()
        temp = np.vstack((temp, np.zeros((MAX_COL_COUNT - n, temp.shape[1])))).astype('float')
        temp = np.nan_to_num(temp)
        feature_dic[f_g] = torch.FloatTensor(temp)

    mask = torch.tensor([1]*n + [0]*(MAX_COL_COUNT - n), dtype=torch.uint8)
    return feature_dic, mask.view(1, MAX_COL_COUNT)

# -------------------------
# Prediction Pipeline
# -------------------------
def predict_table(df):
    feature_dic, mask = extract(df)
    emissions = classifier(feature_dic).view(1, MAX_COL_COUNT, -1)
    predictions = model.decode(emissions, mask=mask)
    predicted_types = label_enc.inverse_transform(predictions[0][:df.shape[1]])
    return list(df.columns), predicted_types

# -------------------------
# Read CSV with Encoding Fallback
# -------------------------
def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")

# -------------------------
# Main Entry Point
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Run SATO predictions on one CSV file or a folder of CSVs.")
    parser.add_argument("--csv_file", type=str, help="Path to input CSV file.")
    parser.add_argument("--csv_folder", type=str, help="Path to a folder containing multiple CSV files.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to write output JSON.")
    args = parser.parse_args()

    all_results = []

    if args.csv_file:
        if not os.path.exists(args.csv_file):
            print(f"Error: File '{args.csv_file}' not found.")
            return
        try:
            df = safe_read_csv(args.csv_file)
            df.to_csv("temp_table.tsv", sep='\t', index=False)
            print(f"Converted {args.csv_file} to TSV (temp_table.tsv)")
            col_names, predictions = predict_table(df)
            all_results.append((args.csv_file, col_names, predictions))
        except Exception as e:
            print(f"[ERROR] Failed to process {args.csv_file}: {e}")
            return

    elif args.csv_folder:
        if not os.path.isdir(args.csv_folder):
            print(f"Error: Folder '{args.csv_folder}' not found.")
            return
        for file_name in os.listdir(args.csv_folder):
            if file_name.endswith(".csv"):
                file_path = os.path.join(args.csv_folder, file_name)
                try:
                    df_multi = safe_read_csv(file_path)
                    if df_multi.empty or df_multi.shape[1] == 0:
                        print(f"[SKIPPED] {file_path} is empty or has no columns.")
                        continue
                    df_multi.to_csv("temp_table.tsv", sep='\t', index=False)
                    print(f"Converted {file_path} to TSV (temp_table.tsv)")
                    col_names, predictions = predict_table(df_multi)
                    all_results.append((file_path, col_names, predictions))
                except Exception as e:
                    print(f"[ERROR] Failed to process {file_path}: {e}")
    else:
        print("Error: Provide either --csv_file or --csv_folder")
        return

    for table_path, col_names, predictions in all_results:
        print(f"\nPredicted Semantic Types for: {table_path}")
        for col, typ in zip(col_names, predictions):
            print(f"  {col}: {typ}")

    if args.output_json:
        import json
        export_data = []
        for table_path, col_names, predictions in all_results:
            export_data.append({
                "file": table_path,
                "columns": [{"column_name": col, "predicted_type": typ} for col, typ in zip(col_names, predictions)]
            })
        with open(args.output_json, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"\nSaved predictions to: {args.output_json}")

if __name__ == "__main__":
    main()
