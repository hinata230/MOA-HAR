model_configs = {
    "LSTM" : {
            'name' : 'LSTM',
            'bidir' : False,
            'clip_val' : 10,
            'drop_prob' : 0.5,
            'n_epochs_hold' : 120,
            'n_layers' : 2,
            'learning_rate' : [0.0015],
            'weight_decay' : 0.001,
            'n_residual_layers' : 0,
            'n_highway_layers' : 0,
            'diag' : 'Architecure chosen is baseline LSTM with 1 layer',
            'save_file' : 'results_lstm1.txt'
    },

    "SC" : {
            'name' : 'SC',
            'bidir' : False,
            'clip_val' : 10,
            'drop_prob' : 0.5,
            'n_epochs_hold' : 120,
            'n_layers' : 2,
            'learning_rate' : [0.0015],
            'weight_decay' : 0.001,
            'n_residual_layers' : 2,
            'n_highway_layers' : 2,
            'diag' : 'Architecure chosen is Softmax Logistic Regression Model with 2 layers',
            'save_file' : 'results_res_logistic.txt'
    }
}

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

# Output classes to learn how to classify
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

TRAIN = "train/"
TEST = "test/"
DATASET_PATH = "dataset/UCI_dataset/"

X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"


n_classes = 6
n_input = 9
n_hidden = 32
batch_size = 64
n_epochs = 120
drop_prob = 0.5
