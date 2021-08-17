import torch
import torch.nn as nn
import torch.nn.functional as F
import sigopt
import argparse
import dgl
from rgcn_hetero_ns import (EntityClassify, RelGraphEmbedding, train,
                                   validate)
from utils import process_dataset
import shutil
import numpy
import matplotlib.pyplot as plt
from time import time
import os
import itertools

def get_data():
  dataset = process_dataset('ogbn-mag', './dataset')
  hg = dataset[0]
  predict_category = dataset.predict_category
  in_feats = 128
  out_feats = dataset.num_classes
  return dataset, hg, in_feats, out_feats, predict_category

def do_sigopt_run(args=None):
        
  ### hardware
  # sigopt.log_metadata("Machine type", args.instance_type)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  ### dataset
#   sigopt.log_dataset(name=f'OGBN mag - mini batch size {args.batch_size}')
  dataset, hg, in_feats, out_feats, predict_category = get_data()
  train_idx = torch.nonzero(
      hg.nodes[predict_category].data['train_mask'], as_tuple=True)[0]
  valid_idx = torch.nonzero(
      hg.nodes[predict_category].data['valid_mask'], as_tuple=True)[0]
  test_idx = torch.nonzero(
      hg.nodes[predict_category].data['test_mask'], as_tuple=True)[0]
  labels = hg.nodes[predict_category].data['labels']  
  num_nodes = {}
  node_feats = {}  
  for ntype in hg.ntypes:
      num_nodes[ntype] = hg.num_nodes(ntype)
      node_feats[ntype] = hg.nodes[ntype].data.get('feat')  
  ### hyperparameters
  activation = F.relu
#   sigopt.params.setdefaults(dict(
#       num_epochs = args.number_of_epochs,
#       number_of_bases = args.number_of_bases,
#       number_of_layers  = args.number_of_layers,
#       activation = activation.__name__,
#       dropout = args.dropout,
#       self_loop = "True",
#       node_feats_projection = "False",
#       batch_size = args.batch_size,
#       num_workers = args.number_of_workers,
#       embedding_lr = args.embedding_learning_rate,
#       model_lr = args.model_learning_rate,
#       hidden_features = args.hidden_features_layer_1
#   ))
  fanouts = [args.fanout_layer_1, 
             args.fanout_layer_2, 
              args.fanout_layer_3][:args.number_of_layers]
            #  args.fanout_layer_3][:int(sigopt.params.number_of_layers)]
  for i in range(args.number_of_layers):
      # sigopt.params.setdefault(f'hidden_layer_{i+1}_fanout', fanouts[i])
      fanouts[i] = 8 # sigopt.params[f'hidden_layer_{i+1}_fanout']
  
  ### dataloader
  sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=fanouts)
  train_dataloader = dgl.dataloading.NodeDataLoader(
      hg,
      {predict_category: train_idx},
      sampler,
      batch_size=32,#sigopt.params.batch_size,
      shuffle=True,
      drop_last=False,
      num_workers=1,#sigopt.params.num_workers,
  )  
  ### instantiate model
  # sigopt.log_model('Heterogeneous RGCN')
  embedding_layer = RelGraphEmbedding(
      hg,
      in_feats,
      num_nodes,
      node_feats,
      False,# bool(sigopt.params.node_feats_projection),
  )
  # embedding = embedding_layer()
  model = EntityClassify(
      hg,
      in_feats,
      32,#sigopt.params.hidden_features,
      out_feats,
      2,# sigopt.params.number_of_bases,
      2, #sigopt.params.number_of_layers,
      activation,
      .3, #sigopt.params.dropout,
      True, #bool(sigopt.params.self_loop),
  )  
  loss_function = nn.CrossEntropyLoss().to(device)  
#   if bool(sigopt.params.node_feats_projection):
  if False:
    all_parameters = itertools.chain(model.parameters(), embedding_layer.embeddings.parameters())
    #   model_optimizer = torch.optim.Adam(all_parameters, lr=sigopt.params.model_lr)
    model_optimizer = torch.optim.Adam(all_parameters, lr=1e-3)
  else:
    # model_optimizer = torch.optim.Adam(model.parameters(), lr=sigopt.params.model_lr)  
    model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  

  embedding_optimizer = torch.optim.SparseAdam(list(embedding_layer.node_embeddings.parameters()), lr=1e-2)  
    #   list(embedding_layer.node_embeddings.parameters()), lr=sigopt.params.embedding_lr)  
    # list(embedding_layer.node_embeddings.parameters()), lr=1e-2)  
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
  for epoch in range(1, 1 + args.number_of_epochs):
    train_time, train_loss, train_accuracy = train(
      embedding_layer,
      model,
      device,
      embedding_optimizer,
      model_optimizer,
      loss_function,
      train_dataloader,
      labels,
      predict_category,
    )
    # valid_time, valid_loss, valid_accuracy = validate(
    #     embedding_layer,
    #     model,
    #     loss_function,
    #     valid_idx,
    #     labels,
    #     predict_category,
    # )
    test_time, test_loss, test_accuracy = validate(
        embedding_layer,
        model,
        loss_function,
        test_idx,
        labels,
        predict_category,
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
    #             "test loss": test_loss,
    #         # "learning_rate": epoch_lr # only checkpoint lr for circuit strategy
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
  ax.legend(handles=h1+h2, labels=l1+l2, bbox_to_anchor=(0.5, 1.01), loc="lower center", ncol=2, fancybox=True)
  sigopt.log_image(image=fig, name="convergence plot")
  # sigopt.log_image(image=fig, name="convergence plot")
  plt.show()

def get_cli_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-n", "--name")
  parser.add_argument("-e", "--number_of_epochs", default=20, type=int) 
  parser.add_argument("-sz", "--step_size", default=10, type=int)
  parser.add_argument("-elr", "--embedding_learning_rate", default=0.01, type=float)
  parser.add_argument("-mlr", "--model_learning_rate", default=0.01, type=float)
  parser.add_argument("-nb", "--number_of_bases", default=2, type=int)
  parser.add_argument("-hf1", "--hidden_features_layer_1", default=24, type=int)
  parser.add_argument("-hf2", "--hidden_features_layer_2", default=21, type=int)
  parser.add_argument("-hf3", "--hidden_features_layer_3", default=18, type=int)
  parser.add_argument("-f1", "--fanout_layer_1", default=12, type=int)
  parser.add_argument("-f2", "--fanout_layer_2", default=13, type=int)
  parser.add_argument("-f3", "--fanout_layer_3", default=14, type=int)
  parser.add_argument("-nh", "--number_of_layers", default=2, type=int)
  parser.add_argument("-do", "--dropout", default=.5, type=float)
  parser.add_argument("-b", "--batch_size", default=32, type=int)
  parser.add_argument("-nw", "--number_of_workers", default=1, type=int)
  parser.add_argument("-i", "--instance_type", default="m5.16xlarge", type=str)
  parser.add_argument("-data", "--download_data", default=0, type=int) # 1 to download
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = get_cli_args()
  if args.download_data:
    data_download_command = f'aws s3 cp s3://ogb-mag ./dataset --recursive'
    os.system(data_download_command)
    shutil.move('./dataset/ogbn_mag', './dataset/ogbn_mag_dgl')
  torch.manual_seed(13)
  do_sigopt_run(args)
