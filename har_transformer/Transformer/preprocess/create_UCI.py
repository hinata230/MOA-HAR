import sys
# sys.path.append("../data/long-range-arena-main/lra_benchmarks/matching/")
import numpy as np
import pickle

from UCI_loadDataset import load_X, load_y

INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]


TRAIN = "train/"
TEST = "test/"
DATASET_PATH = "../dataset/UCI_dataset/"

X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial_Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial_Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]


y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"


X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)


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

    with open(f"UCI.{component}.pickle", "wb") as f:
        pickle.dump(ds_list, f)

