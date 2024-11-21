import sys
import numpy as np
import pandas as pd
import pickle
import json
import glob

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from MOA_loadDataset import load_sequence


RANDOM_SEED = 13

# Data preprocessing
TIME_STEP = 100

# Hyperparameters optimized
SEGMENT_TIME_SIZE = 100

sensor_file_list = sorted(glob.glob('../dataset/MOA_dataset/sensor_data_*.json'))
app_file_list = sorted(glob.glob('../dataset/MOA_dataset/app_data_*.json'))

output_file_template = "../dataset/MOA_dataset/processed_data_part_{}.csv"

df_list = []

for idx, (sensor_file_path, app_file_path) in enumerate(zip(sensor_file_list, app_file_list)) :
    print(sensor_file_path, app_file_path)

    with open(sensor_file_path) as f :
        sensor = json.loads(f.read())
        list_tmp = [item['_source'] for item in sensor['foo']]
        if list_tmp :
            df_sensor = pd.DataFrame(list_tmp)

    with open(app_file_path) as f :
        app = json.loads(f.read())
        list_tmp = [item['_source'] for item in app['foo']]
        if list_tmp :
            df_app = pd.DataFrame(list_tmp)

    if not list_tmp :
        continue

    df_raw = pd.json_normalize(df_sensor['raw'])
    df_sensor = pd.concat([df_sensor.drop(columns=['raw']), df_raw], axis = 1)
    df_sensor['event_time'] = df_sensor['event_time'].astype('Int64')
    
    df_action = pd.json_normalize(df_app['action_tagging'])
    df_app = pd.concat([df_app.drop(columns=['action_tagging']), df_action], axis =1)
    df_app['start_action_time'] = df_app['action_time'].apply(lambda x: x[0]['start']).astype('Int64')
    df_app['end_action_time'] = df_app['action_time'].apply(lambda x: x[0]['end']).astype('Int64')
    df_app = df_app.drop(columns=['action_time'])

    df_result = pd.merge(df_sensor, df_app, how = 'cross')
    df_result = df_result[(df_result['event_time'] >= df_result['start_action_time']) & (df_result['event_time'] <= df_result['end_action_time']) & (df_result['id_x'] == df_result['id_y'])]
    df_result = df_result.drop_duplicates()

    df_result.to_csv(output_file_template.format(idx), index = False)


for idx in range(len(sensor_file_list)) :
    try :
        df_part = pd.read_csv(output_file_template.format(idx))
        df_list.append(df_part)
    except FileNotFoundError :
        print(f"File {output_file_template.format(idx)} not found.")


df_total = pd.concat(df_list, ignore_index = True)

print(df_total['sub_type'].unique())

#data = pd.read_csv(DATASET_PATH,  header=None, names=COLUMN_NAMES)
#
#data['z-axis'] = data['z-axis'].astype(str).str.replace(';', '')
#data = data.dropna()
#
features = df_result[['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
df_result[['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']] = scaled_features
#
## DATA PREPROCESSING
data_convoluted = []
labels = []
data_convoluted, labels = load_sequence(df_total, SEGMENT_TIME_SIZE, TIME_STEP)

#split_ratio = 0.75 # (75%)
#
#sss = StratifiedShuffleSplit(n_splits = 1, test_size = split_ratio, random_state = 42)
#
#for train_index, sample_index in sss.split(data_convoluted, labels) :
#    data_sample = data_convoluted[sample_index]
#    label_sample = labels[sample_index]
#
#data_convoluted = data_sample
#labels = label_sample

labels = np.array(labels).flatten()
unique_labels = np.unique(labels)
print(unique_labels)
label_mapping = {str(label): idx for idx, label in enumerate(unique_labels)}
print(label_mapping)

mapped_labels = np.array([label_mapping[str(label)] for label in labels])

with open("label_mapping.json", "w") as f:
    json.dump(label_mapping, f)

print("Convoluted data shape: ", data_convoluted.shape)
print("Labels shape:", mapped_labels.shape)


## SPLIT INTO TRAINING AND TEST SETS
X_train, X_test, y_train, y_test = train_test_split(data_convoluted, mapped_labels, test_size=0.2, random_state=RANDOM_SEED, stratify = mapped_labels)
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

    with open(f"MOA.{component}.pickle", "wb") as f:
        pickle.dump(ds_list, f)

