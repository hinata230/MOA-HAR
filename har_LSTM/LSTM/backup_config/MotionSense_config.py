###################################
# MotionSense_dataset Configuration
###################################
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

###################################
# Model Hyperparameter
###################################
model_configs = {
    "LSTM" : {
        'name' : 'LSTM',
        'bidir' : False,
        'clip_val' : 20,
        'drop_prob' : 0.5,
        'n_epochs_hold' : 120,
        'n_layers' : 2,
        'learning_rate' : [0.0025],
        'weight_decay' : 0.001,
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
        'n_epochs_hold' : 120,
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

WINDOW_LENGTH = 50
STRIDE_LENGTH = 50

n_classes = 6
n_input = 12
n_hidden = 30
batch_size = 32
n_epochs = 10

