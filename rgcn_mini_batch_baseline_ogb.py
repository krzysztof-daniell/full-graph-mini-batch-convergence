import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rgcn_ns import EntityClassify, train, validate
from utils import process_dataset


if __name__ == '__main__':
    torch.manual_seed(13)

    dataset = process_dataset('ogbn-mag', '/home/ksadowski/datasets')
    hg = dataset[0]
    predict_category = dataset.predict_category

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 20

    train_idx = torch.nonzero(
        hg.nodes[predict_category].data['train_mask'], as_tuple=True)[0]
    valid_idx = torch.nonzero(
        hg.nodes[predict_category].data['valid_mask'], as_tuple=True)[0]
    test_idx = torch.nonzero(
        hg.nodes[predict_category].data['test_mask'], as_tuple=True)[0]

    labels = hg.nodes[predict_category].data['labels']

    hidden_feats = 128
    out_feats = dataset.num_classes
    num_bases = 2
    num_layers = 2
    activation = F.relu
    dropout = 0.7
    self_loop = True
    batch_size = 1024
    num_workers = 4
    fanouts = [30, 30]
    lr = 0.01

    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=fanouts)
    train_dataloader = dgl.dataloading.NodeDataLoader(
        hg,
        {predict_category: train_idx},
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )

    model = EntityClassify(
        hg,
        hidden_feats,
        out_feats,
        num_bases,
        num_layers,
        activation,
        dropout,
        self_loop,
    )

    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_time = 0

    for epoch in range(1, 1 + num_epochs):
        train_time, train_loss, train_accuracy = train(
            model,
            device,
            optimizer,
            loss_function,
            train_dataloader,
            labels,
            predict_category,
        )
        # valid_time, valid_loss, valid_accuracy = validate(
        #     model, loss_function, valid_idx, labels, predict_category)
        test_time, test_loss, test_accuracy = validate(
            model, loss_function, test_idx, labels, predict_category)

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
