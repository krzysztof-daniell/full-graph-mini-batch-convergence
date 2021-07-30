import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gat import GAT, train, validate
from utils import process_dataset

if __name__ == '__main__':
    torch.manual_seed(13)

    dataset = process_dataset('ogbn-products', '/home/ksadowski/datasets')
    g = dataset[0]
    g = g.remove_self_loop().add_self_loop()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 100

    train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    valid_idx = torch.nonzero(g.ndata['valid_mask'], as_tuple=True)[0]
    test_idx = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]

    in_feats = g.ndata['feat'].shape[-1]
    hidden_feats = 120
    out_feats = dataset.num_classes
    num_heads = 4
    num_layers = 3
    activation = F.relu
    feat_dropout = 0
    attention_dropout = 0
    lr = 0.01


    model = GAT(
        in_feats, 
        hidden_feats, 
        out_feats, 
        [num_heads for _ in range(num_layers)], 
        num_layers,
        activation, 
        feat_dropout, 
        attention_dropout,
    )
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_time = 0

    for epoch in range(1, 1 + num_epochs):
        train_time, train_loss, train_accuracy = train(
            model, optimizer, loss_function, g, train_idx)
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
