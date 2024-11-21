###################################
# WISDM_dataset Configuration
###################################


###################################
# Model Hyperparameter
###################################
model_configs = {
    "LSTM" : {
        'name' : 'LSTM',
        'bidir' : False,
        'clip_val' : 50,
        'drop_prob' : 0.8,
        'n_epochs_hold' : 100,
        'n_layers' : 2,
        'learning_rate' : [0.0025],
        'weight_decay' : 0.05,
        'n_residual_layers' : 0,
        'n_highway_layers' : 0,
        'diag' : 'Architecure chosen is baseline LSTM with 1 layer',
        'save_file' : 'results_lstm1.txt'
    },

    "SC" : {
        'name' : 'SC',
        'bidir' : False,
        'clip_val' : 20,
        'drop_prob' : 0.5,
        'n_epochs_hold' : 100,
        'n_layers' : 2,
        'learning_rate' : [0.0025],
        'weight_decay' : 0.001,
        'n_residual_layers' : 2,
        'n_highway_layers' : 2,
        'diag' : 'Architecure chosen is Softmax Logistic Regression Model with 2 layers',
        'save_file' : 'results_SLRM_logistic.txt'
    }
}
RANDOM_SEED = 13

# Data preprocessing
TIME_STEP = 100

# Hyperparameters optimized
SEGMENT_TIME_SIZE = 100

# This will set the values according to that architecture
# This will stay common for all architectures:
n_classes = 28
n_input = 6
n_hidden = 30
batch_size = 32
n_epochs = 10
n_steps = 100


###################################
# DATA INFORMATION
###################################
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

DATASET_PATH = "dataset/WISDM_dataset/WISDM_ar_v1.1_raw.txt"

