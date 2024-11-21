import torch
import json
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from model import SCModel, LSTMModel, init_weights

# Define function to generate batches of a particular size

def extract_batch_size(_train, step, batch_size):
    shape = list(_train.shape)
    shape[0] = batch_size
    batch = np.empty(shape)

    for i in range(batch_size):
        index = ((step - 1) * batch_size + i) % len(_train)
        batch[i] = _train[index]

    return batch


# Define to function to create one-hot encoding of output labels

#def one_hot_vector(y_, n_classes=n_classes):
#    # e.g.:
#    # one_hot(y_=[[2], [0], [5]], n_classes=6):
#    #     return [[0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]
#
#    y_ = y_.reshape(len(y_))
#    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def getLRScheduler(optimizer, epoch, n_epochs_hold, n_epochs_decay):
    def lambdaRule(epoch):
        lr_l = 1.0 - max(0, epoch - n_epochs_hold) / float(n_epochs_decay + 1)
        return lr_l

    schedular = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdaRule)
    #schedular = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return schedular

def plot(x_arg, param_train, param_test, label, lr):
    plt.figure()
    plt.plot(x_arg, param_train, color='blue', label='train')
    plt.plot(x_arg, param_test, color='red', label='test')
    plt.legend()
    if (label == 'accuracy'):
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.title('Training and Test Accuracy', fontsize=20)
        plt.savefig('Accuracy_' + str(epochs) + str(lr) + '.png')
        plt.show()
    elif (label == 'loss'):
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Training and Test Loss', fontsize=20)
        plt.savefig('Loss_' + str(epochs) + str(lr) + '.png')
        plt.show()
    else:
        plt.xlabel('Learning rate', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Training loss and Test loss with learning rate', fontsize=20)
        plt.savefig('Loss_lr_' + str(epochs) + str(lr) + '.png')
        plt.show()

def evaluate(model_path, net, X_test, y_test, criterion, *, args):

    best_fold_acc = 0
    best_fold_precision = 0
    best_fold_recall = 0
    best_fold_cm = None

    if args.data == 'UCI' :
        from config import UCI_config as cfg

    elif args.data == 'WISDM' :
        from config import WISDM_config as cfg

    elif args.data == 'MotionSense' :
        from config import MotionSense_config as cfg

    elif args.data == 'MOA' :
        from config import MOA_config as cfg
        
    model_params = cfg.model_configs.get(args.model)

    LABELS = cfg.LABELS
    n_classes = cfg.n_classes
    n_epochs_hold = model_params['n_epochs_hold']
    n_epochs_decay = cfg.batch_size - n_epochs_hold
    epochs = cfg.n_epochs

    NAME = args.model
    test_batch = len(X_test)
    n_steps = len(X_test[0])  # 128 timesteps per series

    if NAME == 'LSTM':
        net = LSTMModel(
                n_input = cfg.n_input,
                n_hidden = cfg.n_hidden,
                n_layers = model_params['n_layers'],
                n_classes = cfg.n_classes,
                drop_prob = model_params['drop_prob'],
                n_highway_layers = model_params['n_highway_layers']
        )
    elif NAME == 'SC' :
        net = SCModel(
                n_input = cfg.n_input,
                n_classes = cfg.n_classes,
                n_seq_len = n_steps
        )


    all_fold_prob = []

    for path in model_path :
        checkpoint = torch.load(path)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        net.to(device)
        net.load_state_dict(checkpoint["model_state_dict"])
        params = checkpoint["params"]
        net.eval()

        inputs, targets = torch.from_numpy(X_test), torch.from_numpy(y_test.flatten('F'))

        if NAME != 'SC' :
            test_h = net.init_hidden(test_batch)

        if (torch.cuda.is_available() ):
                inputs, targets = inputs.cuda(), targets.cuda()

        if NAME != 'SC' :
            test_h = tuple([each.data for each in test_h])
            output = net(inputs.float(), test_h)

        elif NAME == 'SC' :
            output = net(inputs.reshape(test_batch, -1).float()) 

#        prob = torch.softmax(output, dim = 1)
#        all_fold_prob.append(prob.cpu().detach().numpy())

    test_loss = criterion(output, targets.long())
    top_p, top_class = output.topk(1, dim=1)
    targets = targets.view(*top_class.shape).long()
    equals = top_class == targets

    if (torch.cuda.is_available()) :
        top_class, targets = top_class.cpu(), targets.cpu()

    test_accuracy = torch.mean(equals.type(torch.FloatTensor))
    test_f1score = metrics.f1_score(top_class, targets, average='macro')


#    print("Final loss is: {}".format(test_loss.item()))
    print("Final accuracy is: {}". format(test_accuracy))
    print("Final f1 score is: {}".format(test_f1score))

#    if torch.cuda.is_available():
#        targets = targets.cpu()
#
#    pred = np.argmax(prob.cpu().detach().numpy(), axis = 1)
#    fold_accuracy = accuracy_score(targets.numpy(), pred)
#
#    avg_prob = np.mean(all_fold_prob, axis = 0)
#    final_pred = np.argmax(avg_prob, axis = 1)
#
##    test_accuracy = torch.mean(equals.type(torch.FloatTensor))
##    test_f1score = metrics.f1_score(top_class, targets, average='macro')
#
#    final_accuracy = accuracy_score(targets.numpy(), final_pred)
#    final_f1score = metrics.f1_score(targets.numpy(), final_pred, average='macro')
#
#    if fold_accuracy > best_fold_acc :
#        best_fold_acc = fold_accuracy
#        best_fold_precision = precision_score(targets.numpy(), pred, average = 'macro')
#        best_fold_recall = recall_score(targets.numpy(), pred, average = 'macro')
#        best_fold_cm = confusion_matrix(targets.numpy(), pred)
#
#
#    print("Final accuracy is: {}". format(final_accuracy))
#    print("Final f1 score is: {}". format(final_f1score))

#    print("Best fold accuracy is: {}".format(best_fold_acc))
#    print("Best fold precision is: {}".format(best_fold_precision))
#    print("Best fold recall is: {}".format(best_fold_recall))

#    num_classes = best_fold_cm.shape[0]
#    print(num_classes)
#
#    with open('../../har_transformer/har/preprocess/label_mapping.json', 'r') as f:
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
#    plt.savefig(f"Confusion Matrix {args.data}.png", dpi=300, bbox_inches='tight')

#     confusion_matrix = metrics.confusion_matrix(top_class, targets)
#    print("---------Confusion Matrix--------")
#    print(confusion_matrix)
#    normalized_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100
#    plotConfusionMatrix(normalized_confusion_matrix)

def plotConfusionMatrix(normalized_confusion_matrix):
    plt.figure()
    plt.imshow(
        normalized_confusion_matrix,
        interpolation='nearest',
        cmap=plt.cm.rainbow
    )
    plt.title("Confusion matrix \n(normalised to % of total test data)")
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.png")
    plt.show()

