import numpy as np
import pandas as pd

# Load "X" (the neural network's training and testing inputs)

def load_sequence(data, SEGMENT_TIME_SIZE, TIME_STEP):
    data_convoluted = []
    labels = []

    for i in range(0, len(data) - SEGMENT_TIME_SIZE, TIME_STEP):
        x = data['x-axis'].values[i: i + SEGMENT_TIME_SIZE]
        y = data['y-axis'].values[i: i + SEGMENT_TIME_SIZE]
        z = data['z-axis'].values[i: i + SEGMENT_TIME_SIZE]
        data_convoluted.append([x, y, z])

        # Label for a data window is the label that appears most commonly
        label = np.unique(data['activity'][i: i + SEGMENT_TIME_SIZE], return_counts=True)
        label = label[0][np.argmax(label[1])]
        labels.append(label)

    # Convert to numpy
    data_convoluted = np.asarray(data_convoluted, dtype=np.float32).transpose(0, 2, 1)

    # One-hot encoding
    labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
    labels = np.argmax(labels, axis=1)
#    labels = np.argmax(labels, axis=1).reshape(-1, 1)

    return data_convoluted, labels
