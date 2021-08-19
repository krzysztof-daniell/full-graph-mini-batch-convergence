import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graphsage import GraphSAGE, train_mini_batch, validate
from utils import process_dataset

if __name__ == '__main__':
    torch.manual_seed(13)

    dataset = process_dataset('ogbn-products', '/home/ksadowski/datasets')
    g = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 30

    train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    valid_idx = torch.nonzero(g.ndata['valid_mask'], as_tuple=True)[0]
    test_idx = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]

    in_feats = g.ndata['feat'].shape[-1]
    hidden_feats = 256
    out_feats = dataset.num_classes
    num_layers = 3
    aggregator_type = 'mean'
    batch_norm = True
    activation = F.relu
    input_dropout = 0.1
    dropout = 0.5
    batch_size = 1000
    num_workers = 4
    fanouts = [5, 10, 15]
    lr = 0.003

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

    model = GraphSAGE(
        in_feats,
        hidden_feats,
        out_feats,
        num_layers,
        aggregator_type,
        batch_norm,
        input_dropout,
        dropout,
        activation,
    )

    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_time = 0

    for epoch in range(1, 1 + num_epochs):
        train_time, train_loss, train_accuracy = train_mini_batch(
            model, device, optimizer, loss_function, train_dataloader)
        # valid_time, valid_loss, valid_accuracy = validate(
        #     model, loss_function, g, valid_idx)
        test_time, test_loss, test_accuracy = validate(
            model, loss_function, g, test_idx)

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
