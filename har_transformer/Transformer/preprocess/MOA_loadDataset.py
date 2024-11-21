import numpy as np
import pandas as pd

# Load "X" (the neural network's training and testing inputs)

def load_sequence(data, SEGMENT_TIME_SIZE, TIME_STEP):
    data_convoluted = []
    labels = []

    for i in range(0, len(data) - SEGMENT_TIME_SIZE, TIME_STEP):
        accx = data['accX'].values[i: i + SEGMENT_TIME_SIZE]
        accy = data['accY'].values[i: i + SEGMENT_TIME_SIZE]
        accz = data['accZ'].values[i: i + SEGMENT_TIME_SIZE]
        gyrox = data['gyroX'].values[i: i + SEGMENT_TIME_SIZE]
        gyroy = data['gyroY'].values[i: i + SEGMENT_TIME_SIZE]
        gyroz = data['gyroZ'].values[i: i + SEGMENT_TIME_SIZE]
        data_convoluted.append([accx,accy,accz,gyrox,gyroy,gyroz])

        # Label for a data window is the label that appears most commonly
        label = np.unique(data['sub_type'][i: i + SEGMENT_TIME_SIZE], return_counts=True)
        label = label[0][np.argmax(label[1])]
        labels.append(label)

    # Convert to numpy
    data_convoluted = np.asarray(data_convoluted, dtype=np.float32).transpose(0, 2, 1)

    # One-hot encoding
#    labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
#    labels = np.argmax(labels, axis=1).reshape(-1, 1)

    return data_convoluted, labels
