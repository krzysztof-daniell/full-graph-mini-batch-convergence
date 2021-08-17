### data processing
from utils.dataset_loader import process_dataset
import numpy 
### modeling
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from gat_custom_ns import GAT, train, validate
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
  dataset = process_dataset('ogbn-arxiv', './dataset')
  g = dataset[0]
  in_feats = g.ndata['feat'].shape[-1]
  out_feats = dataset.num_classes
  g = g.remove_self_loop().add_self_loop()
  return dataset, g, in_feats, out_feats

def do_sigopt_run(args=None):

  ### hardware
  sigopt.log_metadata("Machine type", args.instance_type)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

  ### dataset
  sigopt.log_dataset(name=f'OGBN products - full batch')
  dataset, g, in_feats, out_feats = get_data()
  train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
  valid_idx = torch.nonzero(g.ndata['valid_mask'], as_tuple=True)[0]
  test_idx = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]  

  ### hyperparameters
  fanouts = [5, 10, 15]
  activation = F.relu # TODO: parameterize
  sigopt.params.setdefaults(dict(
      number_of_layers = args.number_of_layers,
      activation = activation.__name__,
      edge_dropout = args.edge_dropout,
      attention_dropout = args.attention_dropout,
      input_dropout = args.input_dropout,
      dropout = args.dropout,
      residual = str(args.residual),
      use_attention_dst = str(args.use_attention_dst),
      allow_zero_in_degree = str(args.allow_zero_in_degree),
      use_symmetric_norm = str(args.use_symmetric_norm),
      number_of_heads = args.number_of_heads,
      number_of_workers = args.number_of_workers, # chg to metadata instead of param ? 
      learning_rate = args.learning_rate,
      number_of_epochs = args.number_of_epochs,
      batch_size = args.batch_size,
      lr = args.learning_rate 
  ))  
  hidden_layers = [args.hidden_features_layer_1, 
                   args.hidden_features_layer_2,
                   args.hidden_features_layer_3][:args.number_of_layers] 
                  #  args.hidden_features_layer_3][:int(sigopt.params.number_of_layers)]
  for i in range(args.number_of_layers):
    #pass
    sigopt.params.setdefault(f'hidden_layer_{i+1}_neurons', hidden_layers[i])
    hidden_layers[i] = sigopt.params[f'hidden_layer_{i+1}_neurons']  
    sigopt.params.setdefault(f'hidden_layer_{i+1}_fanout', fanouts[i])
    # fanouts[i] = sigopt.params[f'hidden_layer_{i+1}_fanout'] 

  ### dataloader
  sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=fanouts)
  train_dataloader = dgl.dataloading.NodeDataLoader(
    g,
    train_idx,
    sampler,
    batch_size=sigopt.params.batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=sigopt.params.number_of_workers,
  )

  ### instantiate model
  model = GAT(
    in_feats,
    0,
    6,#hidden_layers,
    out_feats,
    sigopt.params.number_of_heads,
    sigopt.params.number_of_layers,
    0,
    input_dropout=sigopt.params.input_dropout,
    attn_dropout=sigopt.params.attention_dropout,
    edge_dropout=sigopt.params.edge_dropout,
    dropout=sigopt.params.dropout,
    residual=bool(sigopt.params.residual),
    activation=activation,
    use_attn_dst=bool(sigopt.params.use_attention_dst),
    allow_zero_in_degree=bool(sigopt.params.allow_zero_in_degree),
    use_symmetric_norm=bool(sigopt.params.use_symmetric_norm),
  )
  
  loss_function = nn.CrossEntropyLoss().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=sigopt.params.lr)

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

  ### training loop
  t0 = time()
#   for epoch in range(1, 1 + sigopt.params.number_of_epochs):
  for epoch in range(1, 1 + args.number_of_epochs):
    train_time, train_loss, train_accuracy = train(
      model, 
      device, 
      optimizer, 
      loss_function, 
      train_dataloader
    )
#     # valid_time, valid_loss, valid_accuracy = validate(
#     #     model, loss_function, g, valid_idx)
    test_time, test_loss, test_accuracy = validate(
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

    ### intermediate metrics
    sigopt.log_checkpoint({
        "train accuracy": train_accuracy,
         "test accuracy": test_accuracy,
            "train loss": train_loss,
             "test loss": test_loss
    })
    sigopt.log_metric("best epoch - train accuracy", epoch_train_accuracies[best_accuracy['epoch'] - 1])
    sigopt.log_metric("best epoch - train loss", epoch_train_losses[best_accuracy['epoch'] - 1])
    sigopt.log_metric("best epoch - test loss", epoch_test_losses[best_accuracy['epoch'] - 1])
    sigopt.log_metric("best epoch - test accuracy", best_accuracy['value'])
    sigopt.log_metric("best epoch - epoch", best_accuracy['epoch'])
    sigopt.log_metric(
      "mean epoch training time", 
      value=numpy.mean(epoch_train_times),
      stddev=numpy.std(epoch_train_times)
    )
    sigopt.log_metric(
      "mean epoch testing time",
      value=numpy.mean(epoch_test_times),
      stddev=numpy.std(epoch_test_times)
    )  

#   ### final metrics
  tf = time() 
  total_training_time = tf - t0
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
  h1, l1 = ax.get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()
  ax.legend(handles=h1+h2, labels=l1+l2, bbox_to_anchor=(0.5, 1.01), loc="lower center", ncol=2)
  sigopt.log_image(image=fig, name="convergence plot")  

def get_cli_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-n", "--name")
  parser.add_argument("-e", "--number_of_epochs", default=20, type=int) 
  parser.add_argument("-he", "--number_of_heads", default=3, type=int) 
  parser.add_argument("-bs", "--batch_size", default=1024, type=int) 
  parser.add_argument("-rs", "--residual", default=True, type=bool)
  parser.add_argument("-uad", "--use_attention_dst", default=False, type=bool)
  parser.add_argument("-azd", "--allow_zero_in_degree", default=True, type=bool)
  parser.add_argument("-usn", "--use_symmetric_norm", default=True, type=bool)
  parser.add_argument("-lr", "--learning_rate", default=0.01, type=float)
  parser.add_argument("-hf1", "--hidden_features_layer_1", default=16, type=int)
  parser.add_argument("-hf2", "--hidden_features_layer_2", default=17, type=int)
  parser.add_argument("-hf3", "--hidden_features_layer_3", default=18, type=int)
  parser.add_argument("-nh", "--number_of_layers", default=3, type=int)
  parser.add_argument("-do", "--dropout", default=.5, type=float)
  parser.add_argument("-ido", "--input_dropout", default=.5, type=float)
  parser.add_argument("-edo", "--edge_dropout", default=.5, type=float)
  parser.add_argument("-ado", "--attention_dropout", default=.5, type=float)
  parser.add_argument("-nw", "--number_of_workers", default=1, type=int)
  parser.add_argument("-i", "--instance_type", default="local", type=str)
  parser.add_argument("-data", "--download_data", default=0, type=int) # 1 to download
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = get_cli_args()
  if args.download_data:
    data_download_command = f'aws s3 cp s3://ogb-arxiv ./dataset --recursive'
    os.system(data_download_command)
    shutil.move('./dataset/ogbn_arxiv', './dataset/ogbn_arxiv_dgl')
  torch.manual_seed(13)
  do_sigopt_run(args)

# if __name__ == '__main__':
#   torch.manual_seed(13)
  
#   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#   dataset, g, in_feats, out_feats = get_data()
  
#   num_epochs = 100
  
#   train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
#   valid_idx = torch.nonzero(g.ndata['valid_mask'], as_tuple=True)[0]
#   test_idx = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
  
#   #in_feats = g.ndata['feat'].shape[-1]
#   hidden_feats = 250
#   #out_feats = dataset.num_classes
#   num_heads = 3
#   num_layers = 3
#   activation = F.relu
#   input_dropout = 0.1
#   attn_dropout = 0
#   edge_dropout = 0.1
#   dropout = 0.75
#   residual = True
#   use_attn_dst = False
#   allow_zero_in_degree = True
#   use_symmetric_norm = True
  
#   batch_size = 1024
#   num_workers = 4
#   fanouts = [5, 10, 15]
#   lr = 0.002
  
#   sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=fanouts)
#   train_dataloader = dgl.dataloading.NodeDataLoader(
#     g,
#     train_idx,
#     sampler,
#     batch_size=batch_size,
#     shuffle=True,
#     drop_last=False,
#     num_workers=num_workers,
#   )
  
#   model = GAT(
#     in_feats,
#     0,
#     hidden_feats,
#     out_feats,
#     num_heads,
#     num_layers,
#     0,
#     input_dropout=input_dropout,
#     attn_dropout=attn_dropout,
#     edge_dropout=edge_dropout,
#     dropout=dropout,
#     residual=residual,
#     activation=activation,
#     use_attn_dst=use_attn_dst,
#     allow_zero_in_degree=allow_zero_in_degree,
#     use_symmetric_norm=use_symmetric_norm,
#   )
  
#   loss_function = nn.CrossEntropyLoss().to(device)
#   optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  
#   training_time = 0
  
#   for epoch in range(1, 1 + num_epochs):
#     train_time, train_loss, train_accuracy = train(
#         model, device, optimizer, loss_function, train_dataloader)
#     # valid_time, valid_loss, valid_accuracy = validate(
#     #     model, loss_function, g, valid_idx)
#     test_time, test_loss, test_accuracy = validate(
#         model, loss_function, g, test_idx)

#     training_time += train_time

#     print(
#         f'Epoch: {epoch:03} '
#         f'Train Loss: {train_loss:.2f} '
#         # f'valid Loss: {valid_loss:.2f} '
#         f'Test Loss: {test_loss:.2f} '
#         f'Train Accuracy: {train_accuracy * 100:.2f} % '
#         # f'Valid Accuracy: {valid_accuracy * 100:.2f} % '
#         f'Test Accuracy: {test_accuracy * 100:.2f} % '
#         f'Epoch time: {train_time:.2f} '
#         f'Training time: {training_time:.2f} '
#     )
  
