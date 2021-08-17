### data processing
from utils.dataset_loader import process_dataset
import numpy 
### modeling
import graphsage_mini_batch_short as mini_batch_graph 
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
import shutil

def get_data():
  dataset = process_dataset('ogbn-products', './dataset')
  g = dataset[0]
  in_feats = g.ndata['feat'].shape[-1]
  out_feats = dataset.num_classes
  return dataset, g, in_feats, out_feats

def do_sigopt_run(args=None):

  ### hardware
  # sigopt.log_metadata("VM type", args.instance_type)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

  ### dataset
  # sigopt.log_dataset(name=f'OGBN products - mini batch size {args.batch_size}')
  dataset, g, in_feats, out_feats = get_data()
  train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
  valid_idx = torch.nonzero(g.ndata['valid_mask'], as_tuple=True)[0]
  test_idx = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]  

  ### hyperparameters
  activation = F.relu # TODO: parameterize
#   sigopt.params.setdefaults(dict(
#       number_of_layers = str(args.number_of_layers),
#       activation = activation.__name__,
#       dropout = args.dropout,
#       batch_size = args.batch_size,
#       number_of_workers = args.number_of_workers, # chg to metadata instead of param ? 
#       #base_learning_rate = args.base_learning_rate,
#       learning_rate = args.learning_rate,
#       step_size = args.step_size,
#       number_of_epochs = args.number_of_epochs # 300 originally
#   ))
  # sigopt.params.max_learning_rate = 5 * sigopt.params.base_learning_rate
  # define dependent params with args first 
  # in case of stand alone run, set & log default values from args
  # in case of optimization, overwrite based on sigopt suggestion  
  fanouts = [args.fanout_layer_1, 
             args.fanout_layer_2, 
             args.fanout_layer_3][:2]
            #  args.fanout_layer_3][:int(sigopt.params.number_of_layers)]
  hidden_layers = [args.hidden_features_layer_1, 
                   args.hidden_features_layer_2, 
                   args.hidden_features_layer_3][:2]
                #    args.hidden_features_layer_3][:int(sigopt.params.number_of_layers)]
#   for i in range(args.number_of_layers):
#       sigopt.params.setdefault(f'hidden_layer_{i+1}_fanout', fanouts[i])
#       fanouts[i] = sigopt.params[f'hidden_layer_{i+1}_fanout']
#       sigopt.params.setdefault(f'hidden_layer_{i+1}_neurons', hidden_layers[i])
#       hidden_layers[i] = sigopt.params[f'hidden_layer_{i+1}_neurons']  
  
  ### dataloader
  sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=fanouts)
  train_dataloader = dgl.dataloading.NodeDataLoader(
      g,
      train_idx,
      sampler,
      batch_size=33, #sigopt.params.batch_size,
      shuffle=True,
      drop_last=False,
      num_workers=1#sigopt.params.number_of_workers
  )

  ### instantiate model
  # sigopt.log_model('GraphSAGE')
  model = mini_batch_graph.GraphSAGE(
      in_feats,
      hidden_layers,
      out_feats,
      activation,
      .3,#sigopt.params.dropout
  ).to(device)        
  loss_function = nn.CrossEntropyLoss().to(device)
  optimizer = torch.optim.Adam(
      model.parameters(),
      lr = 0.01, #sigopt.params.learning_rate 
  )
  
  ### logging storage
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

  ### training loop
#   for epoch in range(1, 1 + sigopt.params.number_of_epochs):
  for epoch in range(1, 1 + args.number_of_epochs):

    train_time, train_loss, train_accuracy = mini_batch_graph.train(
        model, 
        device,
        optimizer, 
        loss_function, 
        train_dataloader
    )

    test_time, test_loss, test_accuracy = mini_batch_graph.validate(
        model, 
        loss_function, 
        g, 
        test_idx
    )

    epoch_train_accuracies.append(train_accuracy)
    epoch_test_accuracies.append(test_accuracy)
    epoch_train_losses.append(train_loss if type(train_loss) == float else train_loss.detach().numpy())
    epoch_test_losses.append(test_loss if type(test_loss) == float else test_loss.detach().numpy())
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

    ### early stopping check and best epoch logging
    if best_accuracy is None or test_accuracy > best_accuracy['value']:
        best_accuracy = {'value': test_accuracy, 'epoch': epoch}
        early_stopping_counter = 0
    elif best_accuracy is not None and test_accuracy < best_accuracy['value']:
        early_stopping_counter += 1
        if early_stopping_counter >= 20:
            print("EARLY STOP")

    ### checkpoints
    # sigopt.log_checkpoint({
    #     "train accuracy": train_accuracy,
    #         "test accuracy": test_accuracy,
    #         "train loss": train_loss,
    #             "test loss": test_loss
    # })

    ### intermediate metrics
    # sigopt.log_metric("best epoch - train accuracy", epoch_train_accuracies[best_accuracy['epoch'] - 1])
    # sigopt.log_metric("best epoch - train loss", epoch_train_losses[best_accuracy['epoch'] - 1])
    # sigopt.log_metric("best epoch - test loss", epoch_test_losses[best_accuracy['epoch'] - 1])
    # sigopt.log_metric("best epoch - test accuracy", best_accuracy['value'])
    # sigopt.log_metric("best epoch - epoch", best_accuracy['epoch'])
    # sigopt.log_metric("mean epoch training time", 
    #                     value=numpy.mean(epoch_train_times),
    #                     stddev=numpy.std(epoch_train_times))
    # sigopt.log_metric("mean epoch testing time", 
    #                     value=numpy.mean(epoch_test_times), 
    #                     stddev=numpy.std(epoch_test_times))

  ### final metrics
  tf = time() 
  total_training_time = tf - t0
#   sigopt.log_metric("last train accuracy", train_accuracy)
#   sigopt.log_metric("last train loss", train_loss)
#   sigopt.log_metric("last test loss", test_loss)
#   sigopt.log_metric("last test accuracy", test_accuracy)
#   sigopt.log_metric("best epoch - train accuracy", epoch_train_accuracies[best_accuracy['epoch'] - 1])
#   sigopt.log_metric("best epoch - train loss", epoch_train_losses[best_accuracy['epoch'] - 1])
#   sigopt.log_metric("best epoch - test loss", epoch_test_losses[best_accuracy['epoch'] - 1])
#   sigopt.log_metric("best epoch - test accuracy", best_accuracy['value'])
#   sigopt.log_metric("best epoch", best_accuracy['epoch'])
#   sigopt.log_metric("mean epoch training time", 
#                     value=numpy.mean(epoch_train_times),
#                     stddev=numpy.std(epoch_train_times))
#   sigopt.log_metric("mean epoch testing time", 
#                     value=numpy.mean(epoch_test_times), 
#                     stddev=numpy.std(epoch_test_times))
#   sigopt.log_metric("total training time", total_training_time)  

  ### convergence plot
  fig, ax = plt.subplots(1,1)
  ax.plot(numpy.array(epoch_train_accuracies), 'b', label='Train Accuracy')
  ax.plot(numpy.array(epoch_test_accuracies), 'r', label='Test Accuracy')
  ax.set_ylabel('Accuracy', color='black')
  ax2 = ax.twinx()
  ax2.plot(numpy.array(epoch_train_losses), 'b--', label='Train Loss')
  ax2.plot(numpy.array(epoch_test_losses), 'r--', label='Test Loss')
  ax2.set_ylabel('Loss', color='black')
  h1, l1 = ax.get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()
  ax.legend(handles=h1+h2, labels=l1+l2, bbox_to_anchor=(0.5, 1.01), loc="lower center", ncol=2)
  sigopt.log_image(image=fig, name="convergence plot")  
  # sigopt.log_image(image=fig, name="convergence plot")

def get_cli_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-n", "--name")
  parser.add_argument("-e", "--number_of_epochs", default=20, type=int) 
  parser.add_argument("-sz", "--step_size", default=10, type=int)
  parser.add_argument("-lr", "--learning_rate", default=0.01, type=float)
  parser.add_argument("-hf1", "--hidden_features_layer_1", default=16, type=int)
  parser.add_argument("-hf2", "--hidden_features_layer_2", default=17, type=int)
  parser.add_argument("-hf3", "--hidden_features_layer_3", default=18, type=int)
  parser.add_argument("-f1", "--fanout_layer_1", default=12, type=int)
  parser.add_argument("-f2", "--fanout_layer_2", default=13, type=int)
  parser.add_argument("-f3", "--fanout_layer_3", default=14, type=int)
  parser.add_argument("-nh", "--number_of_layers", default=2, type=int)
  parser.add_argument("-do", "--dropout", default=.5, type=float)
  parser.add_argument("-b", "--batch_size", default=1024, type=int)
  parser.add_argument("-nw", "--number_of_workers", default=1, type=int)
  parser.add_argument("-i", "--instance_type", default="m5.16xlarge", type=str)
  parser.add_argument("-data", "--download_data", default=0, type=int) # 1 to download
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = get_cli_args()
  if args.download_data:
    data_download_command = 'aws s3 cp s3://ogb-products ./dataset --recursive'
    os.system(data_download_command)
    shutil.move('./dataset/ogbn_products', './dataset/ogbn_products_dgl')
  torch.manual_seed(13)
  do_sigopt_run(args)