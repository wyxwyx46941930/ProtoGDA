import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool, global_add_pool, global_max_pool


class ImprovedFeatureProjection(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        hidden1 = max(input_dim, output_dim) * 2
        hidden2 = max(input_dim, output_dim)

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7), 

            nn.Linear(hidden2, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.projection(x)


class GNNBackbone(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 3,
                 gnn_type: str = 'GCN',
                 dropout: float = 0.5,
                 pooling: str = 'mean',
                 use_batch_norm: bool = True,
                 use_feature_projection: bool = False,
                 target_input_dim: int = None):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        self.gnn_type = gnn_type
        self.use_feature_projection = use_feature_projection
        self.input_dim = input_dim
        self.target_input_dim = target_input_dim

        if use_feature_projection and target_input_dim is not None and target_input_dim != input_dim:
            self.feature_projection = ImprovedFeatureProjection(
                target_input_dim,
                input_dim,
                dropout=0.3
            )
        else:
            self.feature_projection = None

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim

            if gnn_type == 'GCN':
                conv = GCNConv(in_dim, out_dim)
            elif gnn_type == 'GIN':
                mlp = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim) if use_batch_norm else nn.Identity(),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim)
                )
                conv = GINConv(mlp, train_eps=True)
            elif gnn_type == 'GraphSAGE':
                conv = SAGEConv(in_dim, out_dim)
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")

            self.convs.append(conv)

            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(out_dim))

        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch, edge_weight=None):

        if self.feature_projection is not None and x.shape[1] == self.target_input_dim:
            x = self.feature_projection(x)  

        if edge_weight is not None and edge_weight.dim() > 1:
            edge_weight = edge_weight.squeeze(-1)

        for i in range(self.num_layers):
            if self.gnn_type == 'GCN' and edge_weight is not None:
                x = self.convs[i](x, edge_index, edge_weight)
            else:
                x = self.convs[i](x, edge_index)

            if self.batch_norms is not None:
                x = self.batch_norms[i](x)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batch_norms is not None:
            for bn in self.batch_norms:
                bn.reset_parameters()
        self.classifier.reset_parameters()
