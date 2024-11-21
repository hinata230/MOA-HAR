import sys
# sys.path.append("../data/long-range-arena-main/lra_benchmarks/matching/")
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import MotionSense_loadDataset as mld

FEATURES = [
    "attitude.roll",
    "attitude.pitch",
    "attitude.yaw",
    "gravity.x",
    "gravity.y",
    "gravity.z",
    "rotationRate.x",
    "rotationRate.y",
    "rotationRate.z",
    "userAcceleration.x",
    "userAcceleration.y",
    "userAcceleration.z"
]

# Output classes to learn how to classify
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]
RANDOM_SEED = 13

WINDOW_LENGTH = 50
STRIDE_LENGTH = 50


subject_data_file = '../dataset/MotionSense_dataset/data_subjects_info.csv'
data_dir = '../dataset/MotionSense_dataset/'

subject_data_frame = pd.DataFrame(pd.read_csv(subject_data_file, encoding = "utf-8"))
all_dataset_paths = mld.get_all_dataset_paths(data_dir)
data_frame = mld.load_whole_dataframe_from_paths(all_dataset_paths, subject_data_frame)
data_frame = mld.make_labels(data_frame)

features = data_frame[FEATURES]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
data_frame[FEATURES] = scaled_features

data_convoluted = []
labels = []
data_convoluted, labels = mld.load_sequence(data_frame, WINDOW_LENGTH, STRIDE_LENGTH)

X_train, X_test, y_train, y_test = train_test_split(data_convoluted, labels, test_size=0.2)
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
            "label":inputs[1][0]
        })

        if idx % 100 == 0:
            print(f"{idx}\t\t", end = "\r")

    with open(f"MOTIONSENSE.{component}.pickle", "wb") as f:
        pickle.dump(ds_list, f)

