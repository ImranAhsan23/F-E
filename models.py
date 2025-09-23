# models.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class _TrainMixin:
    def fit(self, data, epochs, train_loader, val_loader):
        device = next(self.parameters()).device
        self.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                self.optimizer.zero_grad()
                out = self(batch.x, batch.edge_index)  # log-softmax
                if hasattr(batch, 'y') and batch.y is not None:
                    loss = self.criterion(out, batch.y)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

class GCN(torch.nn.Module, _TrainMixin):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.g1 = GCNConv(dim_in, dim_h)
        self.g2 = GCNConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, x, edge_index):
        x = self.g1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.g2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphSAGE(torch.nn.Module, _TrainMixin):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.s1 = SAGEConv(dim_in, dim_h)
        self.s2 = SAGEConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, x, edge_index):
        x = self.s1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.s2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module, _TrainMixin):
    def __init__(self, dim_in, dim_h, dim_out, heads=4, dropout=0.6):
        super().__init__()
        self.dropout = dropout
        self.g1 = GATConv(dim_in, dim_h, heads=heads, dropout=dropout)
        self.g2 = GATConv(dim_h * heads, dim_out, heads=1, concat=False, dropout=dropout)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.g1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.g2(x, edge_index)
        return F.log_softmax(x, dim=1)

def get_model(backbone: str, dim_in: int, dim_h: int, dim_out: int):
    name = (backbone or "gcn").lower()
    if name == "gcn": return GCN(dim_in, dim_h, dim_out)
    if name in ["graphsage","sage","gsage"]: return GraphSAGE(dim_in, dim_h, dim_out)
    if name == "gat": return GAT(dim_in, dim_h, dim_out)
    raise ValueError(f"Unknown backbone: {backbone}")

def infer_backbone(model) -> str:
    from inspect import isclass
    if isinstance(model, GCN): return "gcn"
    if isinstance(model, GraphSAGE): return "graphsage"
    if isinstance(model, GAT): return "gat"
    return "gcn"
