import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rgcn_homo_ns import (EntityClassify, RelGraphEmbedLayer, train,
                                 validate, validate_full)
from utils import process_dataset

if __name__ == '__main__':
    torch.manual_seed(13)

    dataset = process_dataset('ogbn-mag', '/home/ksadowski/datasets')
    hg = dataset[0]
    predict_category = dataset.predict_category

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 20

    train_idx = torch.nonzero(
        hg.nodes[predict_category].data.pop('train_mask'), as_tuple=True)[0]
    valid_idx = torch.nonzero(
        hg.nodes[predict_category].data.pop('valid_mask'), as_tuple=True)[0]
    test_idx = torch.nonzero(
        hg.nodes[predict_category].data.pop('test_mask'), as_tuple=True)[0]

    labels = hg.nodes[predict_category].data.pop('labels')
    num_ntypes = len(hg.ntypes)
    num_rels = len(hg.canonical_etypes)

    node_feats = []

    for ntype in hg.ntypes:
        if not len(hg.nodes[ntype].data) or 'feat' not in hg.nodes[ntype].data:
            node_feats.append(None)
        else:
            feat = hg.nodes[ntype].data.pop('feat')

            node_feats.append(feat.share_memory_())

    for i, ntype in enumerate(hg.ntypes):
        if ntype == predict_category:
            predict_category_id = i

    g = dgl.to_homogeneous(hg)
    u, v, eid = g.all_edges(form='all')

    _, inverse_index, counts = torch.unique(
        v, return_inverse=True, return_counts=True)
    degrees = counts[inverse_index]
    norm = torch.ones(eid.shape[0]) / degrees
    norm = norm.unsqueeze(dim=1)

    g.edata['norm'] = norm
    g.edata['etype'] = g.edata[dgl.ETYPE]
    g.ndata['type_id'] = g.ndata[dgl.NID]
    g.ndata['ntype'] = g.ndata[dgl.NTYPE]

    node_ids = torch.arange(g.num_nodes())
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == predict_category_id)
    target_node_ids = node_ids[loc]

    g = g.formats('csc')

    hidden_feats = 128
    out_feats = dataset.num_classes
    num_bases = 2
    num_layers = 2
    activation = F.relu
    dropout = 0.7
    self_loop = True
    layer_norm = False
    batch_size = 1024
    num_workers = 4
    fanouts = [30, 30]
    embedding_lr = 0.08
    model_lr = 0.01

    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=fanouts)
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        target_node_ids[train_idx],
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    test_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        target_node_ids[test_idx],
        sampler,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )

    embedding_layer = RelGraphEmbedLayer(
        node_tids,
        num_ntypes,
        node_feats,
        hidden_feats,
    )
    model = EntityClassify(
        hidden_feats,
        out_feats,
        num_rels,
        num_layers,
        num_bases,
        activation,
        dropout,
        self_loop,
        layer_norm,
    )

    loss_function = nn.CrossEntropyLoss().to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=model_lr)
    embedding_optimizer = torch.optim.SparseAdam(
        embedding_layer.node_embeddings.parameters(), lr=embedding_lr)

    training_time = 0

    for epoch in range(1, 1 + num_epochs):
        train_time, train_loss, train_accuracy = train(
            embedding_layer,
            model,
            embedding_optimizer,
            model_optimizer,
            loss_function,
            node_feats,
            labels,
            train_dataloader,
        )
        # valid_time, valid_loss, valid_accuracy = validate(
        #     embedding_layer,
        #     model,
        #     loss_function,
        #     node_feats,
        #     labels,
        #     valid_dataloader,
        # )
        test_time, test_loss, test_accuracy = validate(
            embedding_layer,
            model,
            loss_function,
            node_feats,
            labels,
            test_dataloader,
        )
        # test_time, test_loss, test_accuracy = validate_full(
        #     embedding_layer,
        #     model,
        #     loss_function,
        #     node_feats,
        #     labels,
        #     g,
        #     test_idx,
        # )

        training_time += train_time

        print(
            f'Epoch: {epoch:03} '
            f'Train Loss: {train_loss:.2f} '
            # f'valid Loss: {valid_loss:.2f} '
            f'Test Loss: {test_loss:.2f} '
            f'Train Accuracy: {train_accuracy * 100:.2f} % '
            # f'Valid Accuracy: {valid_accuracy * 100:.2f} % '
            f'Test Accuracy: {test_accuracy * 100:.2f} % '
            f'Training Epoch time: {train_time:.2f} '
            f'Inference Epoch time: {test_time:.2f} '
            f'Training time: {training_time:.2f} '
        )
