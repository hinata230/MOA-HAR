import os
import sys
import logging
import argparse
import random
import math
import json
import time
import itertools
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from utils import redirect_stdout
from config import Config
from models.model_LRA import ModelForSC, ModelForSCDual
from models.dataset_LRA import LRADataset
import pickle
import requests
import json
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
device_ids = list(range(torch.cuda.device_count()))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validation(model, mat_lst, ds_iter, total_step, training_config, model_config, checkpoint_path, writer, task):
    val_acc = []
    eval_losses = AverageMeter()
    eval_attn_losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        for dev_step_idx in range(training_config["num_eval_steps"]):
            batch = next(iter(ds_iter['dev']))
            if task == 'lra-retrieval':
                input_ids_0 = batch['input_ids_0'].cuda()
                mask_0 = batch['mask_0'].cuda()
                input_ids_1 = batch['input_ids_1'].cuda()
                mask_1 = batch['mask_1'].cuda()
                #print(mask[0])
                label = batch['label'].cuda()
                outputs, attn_lst = model(input_ids_0, input_ids_1, mask_0, mask_1, label, mat_lst,False)
                if len(mat_lst) == 0:
                    attn_loss_0 = sum(sum(outputs["attn_loss_0"]))/4*model_config["num_layers"] * training_config["attn_loss_scale"]
                    attn_loss_1 = sum(sum(outputs["attn_loss_1"]))/4*model_config["num_layers"] * training_config["attn_loss_scale"]
                    attn_loss = (attn_loss_0 + attn_loss_1) /2
                else:
                    attn_loss = 0
            else:
                input = batch['input_ids_0'].cuda()
                mask = batch['mask_0'].cuda()
                label = batch['label'].cuda()
                outputs = model(input,mask,label,mat_lst,False)

            loss = outputs["loss"].mean()
            eval_losses.update(loss.mean())
            acc = outputs["accu"].mean()
            val_acc.append(acc)

        total_acc = sum(val_acc) / len(val_acc)
    print("\nValidation Results")
    print("Valid Loss: %2.5f" % eval_losses.avg)
    print("Valid Accuracy: %2.5f" % total_acc)

    return total_acc



def train_step(model, optimizer, lr_scheduler, ds_iter, amp_scaler, training_config, model_config, writer, task, name):


    logger.info("***** Running training *****")
    logger.info("  Total steps = %d", training_config["num_train_steps"])
    losses = AverageMeter()

#    checkpoint_path = training_config['checkpoint_path']
    best_dev_accu = 0

    total_step = training_config["num_train_steps"]

    epoch_iterator = tqdm(ds_iter['train'],
                              desc=f"Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
    
    mat_lst = []
    update = 0
    total_time = 0

    model.train()
    init_t = time.time()
    all_predictions = []
    all_labels = []

    for step, batch in enumerate(epoch_iterator):    
        if (step + 1) % training_config["eval_frequency"] == 0:
            is_attn = True
        else:
            is_attn = False
        if task == 'lra-retrieval':
            input_ids_0 = batch['input_ids_0'].cuda()
            mask_0 = batch['mask_0'].cuda()
            input_ids_1 = batch['input_ids_1'].cuda()
            mask_1 = batch['mask_1'].cuda()
            #print(mask[0])
            label = batch['label'].cuda()
            outputs = model(input_ids_0, input_ids_1, mask_0, mask_1, label, mat_lst, True)
        else:
            input = batch['input_ids_0'].cuda()
            mask = batch['mask_0'].cuda()
            #print(mask[0])
            label = batch['label'].cuda()
            #print(label)
            outputs = model(input, mask, label, mat_lst, True)

        loss = outputs["loss"].mean()
        acc = outputs["accu"].mean()
        
        amp_scaler.scale(loss).backward() # loss.backward()
        amp_scaler.unscale_(optimizer)
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1) # Gradient Clipping
        amp_scaler.step(optimizer)
        amp_scaler.update()
        lr_scheduler.step()
        losses.update(loss)
        epoch_iterator.set_description(
                    f"Training (%d / %d Steps) (loss=%2.5f)" % (step, total_step, losses.val))

        if (step + 2) > total_step :
            break


#    print('total training step (k): {}'.format(total_step/1000.0))
#    print("total training time (s): {}".format(time.time()-init_t))
#    print("total training time (ms): {}".format(total_time))
#    print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak']>>20))
#    print("allocated memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.allocated']>>20))
#    print(torch.cuda.memory_summary(device=device_ids))

    return model, mat_lst


def evaluation(model, mat_lst, ds_iter, training_config, task):

    eval_losses = AverageMeter()
   
    model.load_state_dict(torch.load(f"pretrained/model_with_params_transformer_{task}.pth"))
    model.eval()

    prob_list = []
    pred_list = []
    labels_list = []

    with torch.no_grad():
        for _, batch in ds_iter['test']:
            if task == 'lra-retrieval':
                input_ids_0 = batch['input_ids_0'].cuda()
                mask_0 = batch['mask_0'].cuda()
                input_ids_1 = batch['input_ids_1'].cuda()
                mask_1 = batch['mask_1'].cuda()
                #print(mask[0])
                label = batch['label'].cuda()
                outputs = model(input_ids_0, input_ids_1, mask_0, mask_1, label, mat_lst, False)
                #outputs, attn_lst = model(input_ids_0, input_ids_1, mask_0, mask_1, label, mat_lst, False)
            else:
                input = batch['input_ids_0'].cuda()
                mask = batch['mask_0'].cuda()
                #print(mask[0])
                label = batch['label'].cuda()
                outputs = model(input,mask,label,mat_lst, False)
                #outputs, attn_lst = model(input,mask,label,mat_lst, False)
            loss = outputs["loss"].mean()
            eval_losses.update(loss.mean())

            prob = torch.softmax(outputs["pred"], dim=-1).cpu().numpy()
            prob_list.append(prob)
            pred_list.append(np.argmax(prob, axis = -1))
            labels_list.append(label.cpu().numpy())

    if prob_list :
        prob_list = np.concatenate(prob_list, axis = 0)
        pred_list = np.concatenate(pred_list, axis = 0)
        labels_list = np.concatenate(labels_list, axis = 0)

        accuracy = metrics.accuracy_score(labels_list, pred_list)
        precision = metrics.precision_score(labels_list, pred_list, average='macro')
        recall = metrics.recall_score(labels_list, pred_list, average='macro')
        f1 = metrics.f1_score(labels_list, pred_list, average='macro')


    print("Evaluation Results")
#    print("Loss: %2.5f" % eval_losses.avg)
    print("Accuracy: %2.5f" % accuracy)
    print("Final f1 score is: %2.5f" % f1)

#    with open('preprocess/label_mapping.json', 'r') as f:
#        label_mapping = json.load(f)
#
#    mapped_labels = [key for key, value in sorted(label_mapping.items(), key=lambda item: item[1])]
#
#
#    plt.figure(figsize=(5, 5))
#    sns.heatmap(best_fold_cm,
#            annot=True,
#            fmt="d",
#            cmap="Blues",
#            cbar=True,
#            annot_kws={"size":3},
#            cbar_kws={"shrink":0.75},
#            xticklabels = True, yticklabels = True,
#            linecolor = 'black', linewidth = 0.1
#            )
#    plt.title("Best Fold Confusion Matrix Heatmap", fontsize = 7)
#    plt.xlabel("Predicted Labels", fontsize = 7)
#    plt.ylabel("True Labels", fontsize = 7)
#    plt.gca().xaxis.set_ticks_position('top')
#    plt.gca().xaxis.set_label_position('top')
#    plt.gca().set_xticks(np.arange(num_classes)+0.5)
#    plt.gca().set_yticks(np.arange(num_classes)+0.5)
#    plt.gca().set_xticklabels(mapped_labels, fontsize = 8)
#    plt.gca().set_yticklabels(mapped_labels, fontsize = 8)
#
#    plt.xticks(fontsize = 5)
#    plt.yticks(fontsize = 5)
#    plt.savefig(f"Confusion Matrix MOA.png", dpi=300, bbox_inches='tight')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type = str, default="train",
                        help="train eval")
#    parser.add_argument("--checkpoint", type = str, default="test",
#                        help="load ./checkpoints/model_name.model to evaluation")
    parser.add_argument("--task", type = str, default="lra-image",
                        help = "lra-listops, lra-retrieval, lra-text, lra-pathfinder32-curv_contour_length_14")
    parser.add_argument('--random', type=int, default=0)
    parser.add_argument('--name', type=str)
    parser.add_argument('--dsteps', type=int)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    if args.task == 'lra-pathfinder':
        args.task = 'lra-pathfinder32-curv_contour_length_14'


    ### get model config ###
    model_config = Config[args.task]["model"]
    model_config["mixed_precision"] = True
#    model_config["max_seq_len"] = int(2 ** math.ceil(math.log2(model_config["max_seq_len"])))
    model_config["random_seed"] = args.random
    model_config["task"] = args.task
    training_config = Config[args.task]["training"]
    training_config["num_dense_train_steps"] = args.dsteps
    data_config = Config[args.task]["dataset"]

    ### log preparation ###
#    log_dir = './logs/log-{}/'.format(args.random)
#    if not os.path.exists(log_dir):
#        os.mkdir(log_dir)
#    log_dir = os.path.join(log_dir, args.task)
#    if not os.path.exists(log_dir):
#        os.mkdir(log_dir)
#
#    log_path = os.path.join(log_dir,'{}.{}.log'.format(args.mode, args.name))
#    redirect_stdout(open(log_path, 'w'))
#
#    writer = SummaryWriter(os.path.join(log_dir,'{}.tensorboard'.format(args.name)))
#
    ###  set the random seeds for deterministic results. ####
    SEED = args.random
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device_ids = list(range(torch.cuda.device_count()))
    model_config['batch_size'] = int(training_config['batch_size']/ len(device_ids))
#    print(f"GPU list: {device_ids}")
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    

#    print(json.dumps([model_config, training_config], indent = 4))

    ### model preparation ###
    if args.task == "lra-retrieval":
        model = ModelForSCDual(model_config)
    else:
        model = ModelForSC(model_config)

    model = nn.DataParallel(model, device_ids = device_ids)

#    checkpoint_dir = './checkpoints/checkpoints-{}/'.format(args.random)
#    if not os.path.exists(checkpoint_dir):
#        os.mkdir(checkpoint_dir)
#    checkpoint_dir = os.path.join(checkpoint_dir, args.task)
#    if not os.path.exists(checkpoint_dir):
#        os.mkdir(checkpoint_dir)
#    checkpoint_path = os.path.join(checkpoint_dir, '{}.model'.format(args.name))
#    training_config["checkpoint_path"] = checkpoint_path


    model = model.cuda()
    print(model)
#    print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush = True)
#    print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)

    
    ### data preparation ###

    train_data = DataLoader(LRADataset(f"preprocess/{args.task}.train.pickle", True), batch_size = training_config["batch_size"], drop_last = True)
    test_data = enumerate(DataLoader(LRADataset(f"preprocess/{args.task}.test.pickle", False), batch_size = training_config["batch_size"], drop_last = True))

    if args.task == "lra-retrieval":
        model = ModelForSCDual(model_config)
    else:
        model = ModelForSC(model_config)

    model = nn.DataParallel(model, device_ids = device_ids)

#    checkpoint_dir = './checkpoints/checkpoints-{}/'.format(args.random)
#    if not os.path.exists(checkpoint_dir):
#        os.mkdir(checkpoint_dir)
#    checkpoint_dir = os.path.join(checkpoint_dir, args.task)
#    if not os.path.exists(checkpoint_dir):
#        os.mkdir(checkpoint_dir)
#    checkpoint_path = os.path.join(checkpoint_dir, '{}.model'.format(args.name))
#    training_config["checkpoint_path"] = checkpoint_path

    model = model.cuda()

    ds_iter = {
            "train" : train_data,
            "test" : test_data
    }

    ### training preparation ###

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = training_config["learning_rate"],
        betas = (0.9, 0.999), eps = 1e-6, weight_decay = training_config["weight_decay"]
    )

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer = optimizer,
        max_lr = training_config["learning_rate"],
        pct_start = training_config["warmup"] / training_config["num_train_steps"],
        anneal_strategy = training_config["lr_decay"],
        total_steps = training_config["num_train_steps"]
    )

    amp_scaler = torch.cuda.amp.GradScaler() if model_config["mixed_precision"] else None

    accumu_steps = model_config["bz_rate"] if "bz_rate" in model_config else 1
    print(f"accumu_steps={accumu_steps}")
    training_config['accumu_steps'] = accumu_steps

    ### train ###
    if args.mode == 'train':
        model, mat_lst = train_step(model, optimizer, lr_scheduler, ds_iter, amp_scaler,
                   training_config, model_config, None, args.task, args.name)

       # acc = validation(model, mat_lst, ds_iter, training_config["num_train_steps"], training_config, model_config, checkpoint_path, writer, args.task)

        torch.save(model.state_dict(), f"pretrained/model_with_params_transformer_{args.task}.pth")

    ### eval ###
#    if os.path.exists(checkpoint_path) and checkpoint_path != './checkpoints/test.model':
#        checkpoint = torch.load(checkpoint_path)
#        model.load_state_dict(checkpoint["model_state_dict"])
#        print("loading the best model from: " + checkpoint_path)

    if args.mode == 'eval' :
        evaluation(model, None, ds_iter, training_config, args.task)
    

if __name__ == '__main__':
    main()
