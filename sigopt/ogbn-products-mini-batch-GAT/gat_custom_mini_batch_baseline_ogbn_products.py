import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from gat_custom_ns import GAT, train, validate
from utils import process_dataset

def get_data():
  dataset = process_dataset('ogbn-products', './dataset')
  g = dataset[0]
  g = g.remove_self_loop().add_self_loop()
  in_feats = g.ndata['feat'].shape[-1]
  out_feats = dataset.num_classes 
  return dataset, g, in_feats, out_feats

if __name__ == '__main__':
  torch.manual_seed(13)  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
  dataset, g, in_feats, out_feats = get_data()  
  num_epochs = 100  
  train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
  valid_idx = torch.nonzero(g.ndata['valid_mask'], as_tuple=True)[0]
  test_idx = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]  
  #in_feats = g.ndata['feat'].shape[-1]
  hidden_feats = 4
  #out_feats = dataset.num_classes
  num_heads = 4
  num_layers = 2
  activation = F.relu
  input_dropout = 0.1
  attn_dropout = 0
  edge_dropout = 0.1
  dropout = 0.5
  residual = True
  use_attn_dst = True
  allow_zero_in_degree = True
  use_symmetric_norm = False  
  batch_size = (len(train_idx) + 29) // 30  # 6554
  num_workers = 4
  fanouts = [10, 10]
  lr = 0.01  
  sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=fanouts)
  train_dataloader = dgl.dataloading.NodeDataLoader(
      g,
      train_idx,
      sampler,
      batch_size=batch_size,
      shuffle=True,
      drop_last=False,
      num_workers=num_workers,
  )  
  model = GAT(
      in_feats,
      0,
      hidden_feats,
      out_feats,
      num_heads,
      num_layers,
      0,
      input_dropout=input_dropout,
      attn_dropout=attn_dropout,
      edge_dropout=edge_dropout,
      dropout=dropout,
      residual=residual,
      activation=activation,
      use_attn_dst=use_attn_dst,
      allow_zero_in_degree=allow_zero_in_degree,
      use_symmetric_norm=use_symmetric_norm,
  )
  loss_function = nn.CrossEntropyLoss().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
  training_time = 0  
  for epoch in range(1, 1 + num_epochs):
    print(f"Training epoch {epoch}")
    train_time, train_loss, train_accuracy = train(
      model, 
      device, 
      optimizer, 
      loss_function, 
      train_dataloader
    )
        # valid_time, valid_loss, valid_accuracy = validate(
        #     model, loss_function, g, valid_idx)
    test_time, test_loss, test_accuracy = validate(
      model,
      loss_function, 
      g, 
      test_idx
    )
    training_time += train_time
    print(
      f'Epoch: {epoch:03} '
      f'Train Loss: {train_loss:.2f} '
      # f'valid Loss: {valid_loss:.2f} '
      f'Test Loss: {test_loss:.2f} '
      f'Train Accuracy: {train_accuracy * 100:.2f} % '
      # f'Valid Accuracy: {valid_accuracy * 100:.2f} % '
      f'Test Accuracy: {test_accuracy * 100:.2f} % '
      f'Epoch time: {train_time:.2f} '
      f'Training time: {training_time:.2f} '
    )
