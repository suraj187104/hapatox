"""
GAHT Model Architecture
Ported from 00_MASTER_PIPELINE.ipynb
"""
import torch
import torch.nn as nn

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.data import Data
except ImportError:
    # Fallback if torch_geometric not available
    MessagePassing = object
    Data = object


class EGNNLayer(MessagePassing):
    """E(n)-equivariant Graph Neural Network Layer"""
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='add')
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.SiLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, 1, bias=False)
        )
    
    def forward(self, x, edge_index, pos):
        return self.propagate(edge_index, x=x, pos=pos)
    
    def message(self, x_i, x_j, pos_i, pos_j):
        dist = torch.norm(pos_i - pos_j, dim=-1, keepdim=True)
        edge_feat = self.edge_mlp(torch.cat([x_i, x_j, dist], dim=-1))
        return edge_feat
    
    def update(self, aggr_out, x, pos, edge_index):
        x_new = self.node_mlp(torch.cat([x, aggr_out], dim=-1))
        return x_new


class GAHT(nn.Module):
    """Geo-Aware Hybrid Transformer"""
    def __init__(self, node_feat_dim=9, hidden_dim=128, num_layers=3, n_heads=4):
        super().__init__()
        self.embedding = nn.Linear(node_feat_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, 1)
        )
    
    def forward(self, data):
        x, edge_index, pos, batch = data.x, data.edge_index, data.pos, data.batch
        
        # Embedding
        x = self.embedding(x)
        
        # EGNN layers
        for gnn in self.gnn_layers:
            x = x + gnn(x, edge_index, pos)
        
        # Global pooling by batch
        batch_size = batch.max().item() + 1 if batch is not None else 1
        graph_embeddings = []
        for i in range(batch_size):
            mask = batch == i if batch is not None else torch.ones(x.size(0), dtype=torch.bool)
            graph_embeddings.append(x[mask].mean(dim=0))
        
        graph_repr = torch.stack(graph_embeddings).unsqueeze(0)
        
        # Transformer
        graph_repr = self.transformer(graph_repr).squeeze(0)
        
        # Classification
        out = self.classifier(graph_repr)
        return out.squeeze()
