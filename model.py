import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.utils import degree, add_self_loops
from torch_geometric.nn import GATConv, GINConv, GCNConv


class DirectedGCNConv(torch_geometric.nn.MessagePassing):
    def __init__(self):
        super(DirectedGCNConv, self).__init__(aggr='add')
        self.c = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x, edge_index):
        # edge_index, _ = add_self_loops(edge_index)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[col]

        deg_out = degree(edge_index[0], x.size(0))
        deg_in = degree(edge_index[1], x.size(0))
        degs = deg_out[edge_index[0]] + deg_in[edge_index[1]]
        degs = degs.pow(-1)
        norm = self.c*degs + norm

        x = self.propagate(edge_index, x=x, norm=norm)
        x = F.mish(x)
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class SourceGCNConvEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super(SourceGCNConvEncoder, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.fc2 = torch.nn.Linear(out_channels * 2, out_channels, bias=False)
        self.fc3 = torch.nn.Linear(out_channels * 3, out_channels, bias=False)
        self.conv1 = DirectedGCNConv()
        self.conv2 = DirectedGCNConv()
        # self.conv1 = GATConv(out_channels, out_channels)
        # self.conv2 = GATConv(out_channels, out_channels)
        # self.conv1 = GCNConv(in_channels, out_channels)
        # self.conv2 = GCNConv(out_channels, out_channels)
        # self.conv3 = GCNConv(out_channels,out_channels)

        self.dropout = torch.nn.Dropout(dropout)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        # self.conv3.reset_parameters()

    def forward(self, x, edge_index):
        x_1 = self.fc1(x)
        x_1 = self.dropout(x_1)
        x_2 = self.conv1(x_1, edge_index)
        # x_2 = self.dropout(x_2)
        x_2 = F.mish(x_2)
        x_3 = x_1 * x_2
        x_3 = self.fc2(torch.cat([x_1, x_2], dim=1))
        x_3 = self.dropout(x_3)
        x_4 = self.conv2(x_3, torch.flip(edge_index, [0]))
        # x_4 = self.dropout(x_4)
        x_4 = F.mish(x_4)
        # out = self.fc3(torch.cat([x_1, x_2, x_4], dim=1))
        out = x_1 + x_2 + x_4
        out = F.mish(out)
        # x = self.conv1(x, torch.flip(edge_index, [0]))
        # x = F.mish(x)
        # x = self.conv2(x, edge_index)
        # x = F.mish(x)
        # # x = self.conv3(x, edge_index)
        return out


class TargetGCNConvEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super(TargetGCNConvEncoder, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.fc2 = torch.nn.Linear(out_channels * 2, out_channels, bias=False)
        self.fc3 = torch.nn.Linear(out_channels * 3, out_channels, bias=False)
        self.conv1 = DirectedGCNConv()
        self.conv2 = DirectedGCNConv()
        # self.conv1 = GATConv(out_channels, out_channels)
        # self.conv2 = GATConv(out_channels, out_channels)
        # self.conv1 = GCNConv(in_channels, out_channels)
        # self.conv2 = GCNConv(out_channels, out_channels)
        # self.conv3 = GCNConv(out_channels, out_channels)

        self.dropout = torch.nn.Dropout(dropout)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        # self.conv3.reset_parameters()

    def forward(self, x, edge_index):
        x_1 = self.fc1(x)
        x_1 = self.dropout(x_1)
        x_2 = self.conv1(x_1, torch.flip(edge_index, [0]))
        # x_2 = self.dropout(x_2)
        x_2 = F.mish(x_2)
        x_3 = x_1 * x_2
        x_3 = self.fc2(torch.cat([x_1, x_2], dim=1))
        x_3 = self.dropout(x_3)
        x_4 = self.conv2(x_3, edge_index)
        # x_4 = self.dropout(x_4)
        x_4 = F.mish(x_4)
        # out = self.fc3(torch.cat([x_1, x_2, x_4], dim=1))
        out = x_1 + x_2 + x_4
        out = F.mish(out)
        # x = self.conv1(x, edge_index)
        # x = F.mish(x)
        # x = self.conv2(x, torch.flip(edge_index, [0]))
        # x = F.mish(x)
        # # x = self.conv3(x, torch.flip(edge_index, [0]))
        return out


class DirectedGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.5):
        super(DirectedGCNEncoder, self).__init__()
        self.source_encoder = SourceGCNConvEncoder(in_channels, out_channels, dropout_p)
        self.target_encoder = TargetGCNConvEncoder(in_channels, out_channels, dropout_p)
        # self.layer2 = GCNConvDirected(256, emb_dim)

    def reset_parameters(self):
        self.source_encoder.reset_parameters()
        self.target_encoder.reset_parameters()

    def forward(self, x, edge_index):
        s = self.source_encoder(x, edge_index)
        t = self.target_encoder(x, edge_index)
        return s, t


class EdgeDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, dropout=0.5):
        super(EdgeDecoder, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, z, edge):
        x = z[0][edge[0]] * z[1][edge[1]]
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.mish(x)
        x = self.fc2(x)
        x = self.dropout(x)
        probs = F.sigmoid(x)
        return probs


class DegreeDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, dropout=0.5):
        super(DegreeDecoder, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, z):
        x = z[0] * z[1]
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.mish(x)
        x = self.fc2(x)
        x = self.dropout(x)
        degrees = F.relu(x)
        return degrees


class DMGAE(torch.nn.Module):
    def __init__(self, encoder, edge_decoder, degree_decoder):
        super(DMGAE, self).__init__()
        self.encoder = encoder
        self.edge_decoder = edge_decoder
        self.degree_decoder = degree_decoder

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.edge_decoder.reset_parameters()
        self.degree_decoder.reset_parameters()

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)


class Classifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=1)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc1.reset_parameters()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.mish(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.softmax(x)
        return x
