import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode

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

def get_all_dataset_paths(input_dir) -> []:
    input_files = []
    for dirs, subdirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                input_files.append(os.path.join(dirs, file))
    return input_files



def load_whole_dataframe_from_paths(paths, meta) -> pd.DataFrame:
    df = pd.DataFrame()
    d = {}

    for p in paths:
        if p != 'dataset/MotionSense_dataset/data_subjects_info.csv' :
            c_dir, c_file = p.split('/')[-2], p.split('/')[-1]
            c_cat, c_ses = c_dir.split('_')[-2], c_dir.split('_')[-1]
            c_sub = c_file.split('_')[-1].split('.')[-2]

            if not c_cat in d : 
                d[c_cat] = 0

            elif d[c_cat] < 10 :
                tdf = pd.read_csv(p, encoding = "utf-8")
                tdf = tdf.assign(subject_id = int(c_sub))
                tdf = tdf.assign(session_id = int(c_ses))
                tdf = tdf.assign(category = str(c_cat))
                tdf = tdf.assign(age = int(meta.age[int(c_sub) - 1]))
                tdf = tdf.assign(gender = int(meta.gender[int(c_sub) - 1]))
                tdf = tdf.assign(height = int(meta.height[int(c_sub) - 1]))
                tdf = tdf.assign(weight = int(meta.weight[int(c_sub) - 1]))

                df = pd.concat([df, tdf])

                d[c_cat] = d[c_cat] + 1
        
        df.reset_index(drop=True, inplace=True)
    df.drop(columns = ['Unnamed: 0', 'subject_id', 'session_id', 'age', 'gender', 'height', 'weight'], axis = 1, inplace = True)
    return df


def make_labels(df) :
    lEncoder = LabelEncoder()
    labels = lEncoder.fit(df.category)
    df['label'] = lEncoder.transform(df.category)
    df.drop('category', axis=1, inplace=True)

    return df


def load_sequence(data, SEGMENT_TIME_SIZE, TIME_STEP):
    data_convoluted = []
    labels = []

    for i in range(0, len(data) - SEGMENT_TIME_SIZE, TIME_STEP):
        x = data[FEATURES].values[i: i + SEGMENT_TIME_SIZE]
        data_convoluted.append(x)

        # Label for a data window is the label that appears most commonly
        label = np.unique(data['label'].values[i: i + SEGMENT_TIME_SIZE], return_counts=True)
        label = label[0][np.argmax(label[1])]
        labels.append(label)

    # Convert to numpy

    data_convoluted = np.asarray(data_convoluted, dtype=np.float32)
    labels = np.array(labels).reshape(-1, 1)

    return data_convoluted, labels


