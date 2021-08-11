import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset


class OGBDataset:
    def __init__(
        self,
        g: dgl.DGLGraph,
        num_labels: int,
        predict_category: str = None,
    ) -> None:
        self._g = g
        self._num_labels = num_labels
        self._predict_category = predict_category

    @property
    def num_labels(self):
        return self._num_labels

    @property
    def num_classes(self):
        return self._num_labels

    @property
    def predict_category(self):
        return self._predict_category

    def __getitem__(self, idx: int):
        return self._g

def load_ogbn_products(root: str = None):
    dataset = DglNodePropPredDataset(name='ogbn-products', root=root)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    g, labels = dataset[0]

    labels = labels.squeeze()
    num_labels = len(torch.unique(
        labels[torch.logical_not(torch.isnan(labels))]))

    g.ndata['label'] = labels

    train_mask = torch.zeros((g.num_nodes(),), dtype=torch.bool)
    train_mask[train_idx] = True
    valid_mask = torch.zeros((g.num_nodes(),), dtype=torch.bool)
    valid_mask[valid_idx] = True
    test_mask = torch.zeros((g.num_nodes(),), dtype=torch.bool)
    test_mask[test_idx] = True

    g.ndata['train_mask'] = train_mask
    g.ndata['valid_mask'] = valid_mask
    g.ndata['test_mask'] = test_mask

    ogb_dataset = OGBDataset(g, num_labels)

    return ogb_dataset

def process_dataset(name: str, root: str = None):
    if root is None:
        root = 'datasets'

    if name == 'reddit':
        dataset =  dgl.data.RedditDataset(self_loop=True, raw_dir=root)
    elif name == 'ogbn-products':
        dataset = load_ogbn_products(root=root)

    return dataset