import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rgcn_hetero import (EntityClassify, RelGraphEmbedding, train,
                                validate)
from utils import process_dataset

if __name__ == '__main__':
    torch.manual_seed(13)

    dataset = process_dataset('ogbn-mag', '/home/ksadowski/datasets')
    hg = dataset[0]
    predict_category = dataset.predict_category

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 50

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

    in_feats = hg.nodes[predict_category].data['feat'].shape[-1]
    hidden_feats = 64
    out_feats = dataset.num_classes
    num_bases = 2
    num_layers = 2
    activation = F.relu
    dropout = 0.5
    self_loop = True
    embedding_lr = 0.01
    model_lr = 0.01

    embedding_layer = RelGraphEmbedding(hg, in_feats, num_nodes, node_feats)
    model = EntityClassify(
        hg,
        in_feats,
        hidden_feats,
        out_feats,
        num_bases,
        num_layers,
        activation,
        dropout,
        self_loop,
    )

    loss_function = nn.CrossEntropyLoss().to(device)

    all_parameters = itertools.chain(
        model.parameters(), embedding_layer.embeddings.parameters())
    model_optimizer = torch.optim.Adam(all_parameters, lr=model_lr)
    embedding_optimizer = torch.optim.SparseAdam(
        list(embedding_layer.node_embeddings.parameters()), lr=embedding_lr)

    training_time = 0

    for epoch in range(1, 1 + num_epochs):
        train_time, train_loss, train_accuracy = train(
            embedding_layer,
            model,
            embedding_optimizer,
            model_optimizer,
            loss_function,
            train_idx,
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
