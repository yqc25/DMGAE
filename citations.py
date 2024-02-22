import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data


def load_data_lp_cora(path="./data/cora/", dataset="cora"):
    print('Loading {} dataset...'.format(dataset))
    paper_feature_label = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    papers = paper_feature_label[:, 0].astype(np.int32)
    paper2idx = {k: v for v, k in enumerate(papers)}
    features = sp.csr_matrix(paper_feature_label[:, 1:-1], dtype=np.float32)
    labels = paper_feature_label[:, -1]
    lbl2idx = {k: v for v, k in enumerate(sorted(np.unique(labels)))}
    labels = [lbl2idx[e] for e in labels]

    edges = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.asarray([paper2idx[e] for e in edges.flatten()], np.int32).reshape(edges.shape)

    edges = torch.LongTensor(edges)
    labels = torch.LongTensor(labels)
    features = torch.FloatTensor(features.todense())
    data = data_pyg(edges, features, labels)

    return data


def load_data_lp_cora_ml(path="./data/cora_ml/", dataset="cora_ml"):
    print('Loading {} dataset...'.format(dataset))
    file_name = path + "cora_ml.npz"
    if not file_name.endswith('.npz'):
        file_name += '.npz'

    loader = np.load(file_name, allow_pickle=True)
    loader = dict(loader)
    adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                         loader['adj_indptr']), shape=loader['adj_shape'])

    features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                              loader['attr_indptr']), shape=loader['attr_shape'])

    labels = loader.get('labels')

    x, y = adj.nonzero()
    edges = np.array(list(zip(x, y)))
    edges = torch.LongTensor(edges)
    labels = torch.LongTensor(labels)
    features = torch.FloatTensor(features.todense())
    data = data_pyg(edges, features, labels)

    return data


def load_data_lp_citeseer(path="./data/citeseer/", dataset="citeseer"):
    print('Loading {} dataset...'.format(dataset))
    paper_feature_label = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    papers = paper_feature_label[:, 0]
    paper2idx = {k: v for v, k in enumerate(papers)}
    features = sp.csr_matrix(paper_feature_label[:, 1:-1], dtype=np.float32)
    labels = paper_feature_label[:, -1]
    lbl2idx = {k: v for v, k in enumerate(sorted(np.unique(labels)))}
    labels = [lbl2idx[e] for e in labels]

    edges_ = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.dtype(str))
    edges = []
    for e in edges_:
        if (e[0] in papers) and (e[1] in papers):
            edges.append(e)
    edges = np.array(edges)
    edges = np.asarray([paper2idx[e] for e in edges.flatten()], np.int32).reshape(edges.shape)
    edges = torch.LongTensor(edges)
    labels = torch.LongTensor(labels)
    features = torch.FloatTensor(features.todense())
    data = data_pyg(edges, features, labels)

    return data


def load_data_lp_pubmed(path="./data/pubmed/", dataset="pubmed"):
    print('Loading {} dataset...'.format(dataset))
    paper_feature_lable_file = "Pubmed-Diabetes.NODE.paper.tab"
    citation_file = "Pubmed-Diabetes.DIRECTED.cites.tab"
    paper_feature_lable_file_f = open(path + paper_feature_lable_file, 'r')
    lines = paper_feature_lable_file_f.readlines()
    lst_line_1 = lines[1].split('\t')
    words = []
    for w in lst_line_1:
        if w.startswith('numeric:'):
            words.append(w[8:-4])
    words_dict = {k: v for v, k in enumerate(words)}
    papers = []
    labels = []
    features = np.zeros((19717, 500))
    num_line = -2
    for line in lines:
        line = line.replace('\n', '')
        lst_line = line.split('\t')
        if lst_line[1].startswith('label'):
            papers.append(lst_line[0])
            labels.append(int(lst_line[1][-1]))
            for w in lst_line:
                if w.startswith('w'):
                    lst_w = w.split('=')
                    features[num_line][words_dict[lst_w[0]]] = float(lst_w[1])
        num_line += 1
    papers2idx = {k: v for v, k in enumerate(papers)}
    paper_feature_lable_file_f.close()
    citation_file_f = open(path + citation_file, 'r')
    lines = citation_file_f.readlines()
    lines = lines[2:]
    edges = []
    for line in lines:
        line = line.replace('\n', '')
        lst_line = line.split('\t')
        edges.append([lst_line[1][6:], lst_line[3][6:]])
    edges = np.array(edges)
    edges = np.asarray([papers2idx[e] for e in edges.flatten()], dtype=int).reshape(-1, 2)
    features = sp.csr_matrix(features[:, :], dtype=float)
    edges = torch.LongTensor(edges)
    labels = torch.LongTensor(labels)
    features = torch.FloatTensor(features.todense())
    data = data_pyg(edges, features, labels)
    return data


def data_pyg(edges, features, labels):
    data = Data()
    data.edge_index = edges.t()
    data.x = features
    data.y = labels
    data.num_nodes = features.shape[0]
    data.num_features = features.shape[1]
    data.num_node_features = features.shape[1]
    data.num_edges = edges.shape[0]

    return data


def directed_data(path, dataset):
    path_dataset = path + dataset + '/'
    if dataset == 'Cora':
        return load_data_lp_cora(path_dataset)
    elif dataset == 'Cora_ML':
        return load_data_lp_cora_ml(path_dataset)
    elif dataset == 'CiteSeer':
        return load_data_lp_citeseer(path_dataset)
    elif dataset == 'PubMed':
        return load_data_lp_pubmed(path_dataset)
    else:
        return None


def normalize(mx):
    row_sum = np.array(mx.sum(1))
    r_inv = (row_sum ** -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
