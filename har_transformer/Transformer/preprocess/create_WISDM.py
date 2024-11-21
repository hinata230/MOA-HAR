import sys
# sys.path.append("../data/long-range-arena-main/lra_benchmarks/matching/")
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from WISDM_loadDataset import load_sequence

LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

COLUMN_NAMES = [
    'user',
    'activity',
    'timestamp',
    'x-axis',
    'y-axis',
    'z-axis'
]

RANDOM_SEED = 13

# Data preprocessing
TIME_STEP = 100

# Hyperparameters optimized
SEGMENT_TIME_SIZE = 100

DATASET_PATH = "../dataset/WISDM_dataset/WISDM_ar_v1.1_raw.txt"

data = pd.read_csv(DATASET_PATH,  header=None, names=COLUMN_NAMES)

data['z-axis'] = data['z-axis'].astype(str).str.replace(';', '')
data = data.dropna()

features = data[['x-axis', 'y-axis', 'z-axis']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
data[['x-axis', 'y-axis', 'z-axis']] = scaled_features

# DATA PREPROCESSING
data_convoluted = []
labels = []
data_convoluted, labels = load_sequence(data, SEGMENT_TIME_SIZE, TIME_STEP)

print("Convoluted data shape: ", data_convoluted.shape)
print("Labels shape:", labels.shape)

# SPLIT INTO TRAINING AND TEST SETS
X_train, X_test, y_train, y_test = train_test_split(data_convoluted, labels, test_size=0.2, random_state=RANDOM_SEED)
print("X train size: ", len(X_train))
print("X test size: ", len(X_test))
print("y train size: ", len(y_train))
print("y test size: ", len(y_test))


mapping = {"train": [X_train,  y_train], 
            "test": [X_test, y_test]}


for component in mapping:

    ds_list = []
    for idx, inputs in enumerate(zip(mapping[component][0], mapping[component][1])):
        ds_list.append({
#            "input_ids_0":np.concatenate([inputs[0][0], np.zeros(96, dtype = np.int32)]),
            "input_ids_0":inputs[0],
            "label":inputs[1]
        })

        if idx % 100 == 0:
            print(f"{idx}\t\t", end = "\r")

    with open(f"WISDM.{component}.pickle", "wb") as f:
        pickle.dump(ds_list, f)

