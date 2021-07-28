### data processing
from utils.dataset_loader import process_dataset
import numpy 
### modeling
import graphsage_full_graph_short as full_graph 
import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn 
import torch.nn.functional as F 
### tracking, plotting
import matplotlib.pyplot as plt 
from time import time
### HPO
import sigopt 
### CLI 
import argparse
import os

CIRCUIT_LR = True
def cyclical_learning_rate_policy(epoch=None):
    """
    Cyclical Learning Rates for Training Neural Networks
    https://arxiv.org/pdf/1506.01186.pdf%5D
    """
    cycle = numpy.floor(1 + epoch / (2 * sigopt.params.step_size))
    x = numpy.abs(epoch / sigopt.params.step_size - 2 * cycle + 1)
    learning_rate = sigopt.params.base_learning_rate + \
        (sigopt.params.max_learning_rate - sigopt.params.base_learning_rate) * numpy.maximum(0, 1-x)
    return learning_rate

def get_data():
    dataset = process_dataset('ogbn-products', './dataset')
    g = dataset[0]
    in_feats = g.ndata['feat'].shape[-1]
    out_feats = dataset.num_classes
    return dataset, g, in_feats, out_feats

def do_sigopt_run(args=None):
    ### hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ### dataset
    dataset, g, in_feats, out_feats = get_data()
    train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    valid_idx = torch.nonzero(g.ndata['valid_mask'], as_tuple=True)[0]
    test_idx = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
    ### hyperparameters
    activation = F.relu
    sigopt.params.setdefaults(dict(
        hidden_features = args.hidden_features,
        number_of_layers = args.number_of_layers,
        activation = activation.__name__,
        dropout = args.dropout,
        batch_size = args.batch_size,
        # sigopt.params.number_of_workers = args.number_of_workers # chg to metadata instead of param ? 
        base_learning_rate = args.base_learning_rate,
        max_learning_rate = args.max_learning_rate,
        step_size = args.step_size,
        number_of_epochs = args.number_of_epochs # 300 originally
    ))
    ### instantiate model
    model = full_graph.GraphSAGE(
        in_feats,
        sigopt.params.hidden_features,
        out_feats,
        sigopt.params.number_of_layers,
        activation,
        sigopt.params.dropout
    ).to(device)        
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=sigopt.params.base_learning_rate if CIRCUIT_LR else sigopt.params.learning_rate
    )
    ### logging
    epoch_times = []
    epoch_train_accuracies = []
    epoch_test_accuracies = []
    epoch_train_losses = []
    epoch_test_losses = []
    epoch_train_times = []
    epoch_test_times = []
    best_accuracy = None
    early_stopping_counter = 0
    t0 = time()
    for epoch in range(1, 1 + sigopt.params.number_of_epochs):
        train_time, train_loss, train_accuracy = full_graph.train(
            model, 
            optimizer, 
            loss_function, 
            g, 
            train_idx
        )
        test_time, test_loss, test_accuracy = full_graph.validate(
            model, 
            loss_function, 
            g, 
            test_idx
        )
        epoch_train_accuracies.append(train_accuracy)
        epoch_test_accuracies.append(test_accuracy)
        epoch_train_losses.append(train_loss.detach().numpy())
        epoch_test_losses.append(test_loss.detach().numpy())
        epoch_train_times.append(train_time)
        epoch_test_times.append(test_time)
        print(
            f'Epoch: {epoch:03} '
            f'Train Loss: {train_loss:.2f} '
            f'Test Loss: {test_loss:.2f} '
            f'Train Accuracy: {train_accuracy * 100:.2f} % '
            f'Test Accuracy: {test_accuracy * 100:.2f} % '
            f'Train epoch time: {train_time:.2f} '
        )
        if best_accuracy is None or test_accuracy > best_accuracy['value']:
            best_accuracy = {'value': test_accuracy, 'epoch': epoch}
            early_stopping_counter = 0
        elif best_accuracy is not None and test_accuracy < best_accuracy['value']:
            early_stopping_counter += 1
            if early_stopping_counter >= 20:
                print("EARLY STOP")
        # if test_accuracy >= ACCURACY_THRESHOLD:
        #     if best_num_epochs is None or best_accuracy['epoch'] < best_num_epochs:
        #         best_num_epochs = best_accuracy['epoch']
        #     break
        # if CIRCUIT_LR:
        epoch_lr = None # variable for logging
        for weight_group in optimizer.param_groups:
            epoch_lr = cyclical_learning_rate_policy(epoch=epoch)
            weight_group['lr'] = epoch_lr
        sigopt.log_checkpoint({
            "train accuracy": train_accuracy,
             "test accuracy": test_accuracy,
                "train loss": train_loss,
                 "test loss": test_loss,
             "learning_rate": epoch_lr
        })
        sigopt.log_metric("best epoch - train accuracy", epoch_train_accuracies[best_accuracy['epoch'] - 1])
        sigopt.log_metric("best epoch - train loss", epoch_train_losses[best_accuracy['epoch'] - 1])
        sigopt.log_metric("best epoch - test loss", epoch_test_losses[best_accuracy['epoch'] - 1])
        sigopt.log_metric("best epoch - test accuracy", best_accuracy['value'])
        sigopt.log_metric("best epoch - epoch", best_accuracy['epoch'])
        sigopt.log_metric("mean epoch training time", 
                          value=numpy.mean(epoch_train_times),
                          stddev=numpy.std(epoch_train_times))
        sigopt.log_metric("mean epoch testing time", 
                          value=numpy.mean(epoch_test_times), 
                          stddev=numpy.std(epoch_test_times))
    tf = time() 
    total_training_time = tf - t0
    ### final metrics
    sigopt.log_metric("last train accuracy", train_accuracy)
    sigopt.log_metric("last train loss", train_loss)
    sigopt.log_metric("last test loss", test_loss)
    sigopt.log_metric("last test accuracy", test_accuracy)
    sigopt.log_metric("best epoch - train accuracy", epoch_train_accuracies[best_accuracy['epoch'] - 1])
    sigopt.log_metric("best epoch - train loss", epoch_train_losses[best_accuracy['epoch'] - 1])
    sigopt.log_metric("best epoch - test loss", epoch_test_losses[best_accuracy['epoch'] - 1])
    sigopt.log_metric("best epoch - test accuracy", best_accuracy['value'])
    sigopt.log_metric("best epoch", best_accuracy['epoch'])
    sigopt.log_metric("mean epoch training time", 
                      value=numpy.mean(epoch_train_times),
                      stddev=numpy.std(epoch_train_times))
    sigopt.log_metric("mean epoch testing time", 
                      value=numpy.mean(epoch_test_times), 
                      stddev=numpy.std(epoch_test_times))
    sigopt.log_metric("total training time", total_training_time)
    ### convergence plot
    fig, ax = plt.subplots(1,1)
    ax.plot(numpy.array(epoch_train_accuracies), 'b', label='Train Accuracy')
    ax.plot(numpy.array(epoch_test_accuracies), 'r', label='Test Accuracy')
    ax.set_ylabel('Accuracy', color='black')
    ax2 = ax.twinx()
    ax2.plot(numpy.array(epoch_train_losses), 'b--', label='Train Loss')
    ax2.plot(numpy.array(epoch_test_losses), 'r--', label='Test Loss')
    ax2.set_ylabel('Loss', color='black')
    ax2.legend()
    sigopt.log_image(image=fig, name="convergence plot")

def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name")
    parser.add_argument("-e", "--number_of_epochs", default=100, type=int) 
    parser.add_argument("-sz", "--step_size", default=25, type=int)
    # parser.add_argument("-lr", "--learning_rate", default=0.01, type=float)
    # if CIRCUIT-LR:
    parser.add_argument("-blr", "--base_learning_rate", default=0.005, type=float)
    parser.add_argument("-mlr", "--max_learning_rate", default=0.05, type=float)
    parser.add_argument("-hf", "--hidden_features", default=16, type=int)
    parser.add_argument("-nh", "--number_of_layers", default=2, type=int)
    parser.add_argument("-d", "--dropout", default=.5, type=float)
    parser.add_argument("-b", "--batch_size", default=1024, type=int)
    #parser.add_argument("-nw", "--number_of_workers", default=4, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    data_download_command = 'python download_from_s3.py -b ogb-products -o ./dataset/ -p ogbn_products -f ogbn_products_dgl'
    os.system(data_download_command)
    torch.manual_seed(13)
    do_sigopt_run(get_cli_args())

