import torch
import numpy as np
import pandas as pd
from train import train
from torch import nn
from model import SCModel, LSTMModel, init_weights
from Functions import plot, evaluate
import sys
import csv
import os
import argparse
import random
import json
import glob
from models.dataset_LRA import LRADataset
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

os.makedirs("pretrained", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help='A Type of model')
parser.add_argument("--data", type=str, help='A Dataset to train a model')
parser.add_argument("--mode", type=str, help='train/eval mode')
args = parser.parse_args()

# LSTM Neural Network's internal structure

if args.data == 'UCI' :
    from config import UCI_config as cfg

elif args.data == 'WISDM' :
    from config import WISDM_config as cfg

elif args.data == 'MotionSense' :
    from config import MotionSense_config as cfg

elif args.data == 'MOA' :
    from config import MOA_config as cfg


model_params = cfg.model_configs.get(args.model)

# Training
# check if GPU is available

#train_on_gpu = torch.cuda.is_available()
if (torch.cuda.is_available() ):
    print('Training on GPU')
else:
    print('GPU not available! Training on CPU. Try to keep n_epochs very small')


def main():
    if args.data == 'UCI' :
        # Data file to load X and y values

        from config import UCI_config as udf
        from loadDataset import UCI_loadDataset as uld

        X_train_signals_paths = udf.X_train_signals_paths
        X_test_signals_paths = udf.X_test_signals_paths

        y_train_path = udf.y_train_path
        y_test_path = udf.y_test_path

        X_train = uld.load_X(X_train_signals_paths)
        X_test = uld.load_X(X_test_signals_paths)

        y_train = uld.load_y(y_train_path)
        y_test = uld.load_y(y_test_path)

#        print(X_train)
#        print(y_train)
#
#        print(dict(Counter(np.vstack((y_train, y_test)).flatten())))


    elif args.data == 'WISDM' :
        from config import WISDM_config as wdf
        from loadDataset import WISDM_loadDataset as wld

        data = pd.read_csv(wdf.DATASET_PATH,  header=None, names=wdf.COLUMN_NAMES)
        data['z-axis'] = data['z-axis'].astype(str).str.replace(';', '')
        data = data.dropna()

        features = data[['x-axis', 'y-axis', 'z-axis']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        data[['x-axis', 'y-axis', 'z-axis']] = scaled_features

        data = data.dropna()

#        train_df, test_df = train_test_split(data, test_size = 0.2, stratify=data['activity'], random_state = wdf.RANDOM_SEED)
#
#        X_train, y_train = wld.load_sequence(train_df, wdf.SEGMENT_TIME_SIZE, wdf.TIME_STEP)
#        X_test, y_test = wld.load_sequence(test_df, wdf.SEGMENT_TIME_SIZE, wdf.TIME_STEP)


#        # DATA PREPROCESSING

        data_convoluted = []
        labels = []
        data_convoluted, labels = wld.load_sequence(data, wdf.SEGMENT_TIME_SIZE, wdf.TIME_STEP)

#        print(dict(Counter(labels.flatten())))
#
#        print("Convoluted data shape: ", data_convoluted.shape)
#        print("Labels shape:", labels.shape)

        # SPLIT INTO TRAINING AND TEST SETS
        X_train, X_test, y_train, y_test = train_test_split(data_convoluted, labels, test_size=0.2, random_state=wdf.RANDOM_SEED, stratify = labels)


    elif args.data == 'MotionSense' :
        from loadDataset import MotionSense_loadDataset as mld
        from config import MotionSense_config as mcf

        subject_data_file = 'dataset/MotionSense_dataset/data_subjects_info.csv'
        data_dir = 'dataset/MotionSense_dataset/'

        subject_data_frame = pd.DataFrame(pd.read_csv(subject_data_file, encoding = "utf-8"))
        all_dataset_paths = mld.get_all_dataset_paths(data_dir)
        data_frame = mld.load_whole_dataframe_from_paths(all_dataset_paths, subject_data_frame)
        data_frame = mld.make_labels(data_frame)
        
        features = data_frame[mcf.FEATURES]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        data_frame[mcf.FEATURES] = scaled_features

#        train_df, test_df = train_test_split(data_frame, test_size = 0.2, stratify=data_frame['label'], random_state = mcf.RANDOM_SEED)
#
#        X_train, y_train = mld.load_sequence(train_df, mcf.WINDOW_LENGTH, mcf.STRIDE_LENGTH)
#        X_test, y_test = mld.load_sequence(test_df, mcf.WINDOW_LENGTH, mcf.STRIDE_LENGTH)

        data_convoluted = []
        labels = []
        data_convoluted, labels = mld.load_sequence(data_frame, mcf.WINDOW_LENGTH, mcf.STRIDE_LENGTH)

        X_train, X_test, y_train, y_test = train_test_split(data_convoluted, labels, test_size=0.2,random_state=mcf.RANDOM_SEED, stratify = labels)


#        print("X train size: ", len(X_train))
#        print("X test size: ", len(X_test))
#        print("y train size: ", len(y_train))
#        print("y test size: ", len(y_test))
#

    elif args.data == 'MOA' :
        from loadDataset import MOA_loadDataset as Mld
        from config import MOA_config as Mcf

        X_train = DataLoader(LRADataset(f"../../har_transformer/har/preprocess/MOA.train.pickle", True))
        X_test = DataLoader(LRADataset(f"../../har_transformer/har/preprocess/MOA.test.pickle", False), batch_size = Mcf.batch_size, drop_last = True)


        y_train = X_train.dataset.get_labels()
        y_test = X_test.dataset.get_labels()

        X_train = X_train.dataset.get_x()
        X_test = X_test.dataset.get_x()

#        print(dict(Counter(np.concatenate((y_train, X_test.dataset.get_labels())))))

    
    if args.mode == 'train' :
        training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
        test_data_count = len(X_test)  # 2947 testing series

        if args.data == 'MOA' :
            n_steps = cfg.n_steps
        else :
            n_steps = len(X_train[0])  # 128 timesteps per series
            n_input = len(X_train[0][0])  # 9 input parameters per timestep

        epochs = cfg.n_epochs

        for lr in model_params['learning_rate']:
            if args.model == 'LSTM':
                net = LSTMModel(
                        n_input = cfg.n_input,
                        n_hidden = cfg.n_hidden,
                        n_layers = model_params['n_layers'],
                        n_classes = cfg.n_classes,
                        drop_prob = model_params['drop_prob'],
                        n_highway_layers = model_params['n_highway_layers']
                )
            elif args.model == 'SC' :
                net = SCModel(
                        n_input = cfg.n_input,
                        n_classes = cfg.n_classes,
                        n_seq_len = n_steps
                )
            else:
                print("Incorrect architecture chosen. Please check architecture given in config.py. Program will exit now! :( ")
                sys.exit()
            net.apply(init_weights)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            net = net.float()

            params = train(net, X_train, y_train, X_test, y_test, None, opt=opt, criterion=criterion, epochs=model_params['n_epochs_hold'], clip_val=model_params['clip_val'], args=args)

            save_data = {
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "params": params
            }
            torch.save(save_data, f"pretrained/model_with_params_{args.model}_{args.data}.pth")

#        evaluate(["model.pth"], net, X_test, y_test, criterion, args=args)
#        print(net.state_dict())

    if args.mode == 'eval' :
        criterion = nn.CrossEntropyLoss()
        evaluate([f"pretrained/model_with_params_{args.model}_{args.data}.pth"], None, X_test, y_test, criterion, args=args)


    
main()
