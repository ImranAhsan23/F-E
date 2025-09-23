# unlearning.py
import time
from typing import Optional
import torch
from torch_geometric.datasets import Coauthor, Planetoid
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import subgraph
from models import get_model, infer_backbone

def split_data(data, train_ratio=0.6, val_ratio=0.2):
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    train_cutoff = int(train_ratio * num_nodes)
    val_cutoff = int((train_ratio + val_ratio) * num_nodes)
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[indices[:train_cutoff]] = True
    data.val_mask[indices[train_cutoff:val_cutoff]] = True
    data.test_mask[indices[val_cutoff:]] = True
    return data

def _load_dataset(dataset: str):
    ds = (dataset or "physics").lower()
    if ds in ["cora","citeseer","pubmed"]:
        name_map = {"citeseer":"CiteSeer","pubmed":"PubMed","cora":"Cora"}
        return Planetoid(root='.', name=name_map[ds])
    elif ds in ["cs","physics"]:
        name_map = {"cs":"CS","physics":"Physics"}
        return Coauthor(root='.', name=name_map[ds])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def _ensure_no_grad_features_(data_obj):
    data_obj.x = data_obj.x.detach()
    data_obj.x.requires_grad_(False)
    return data_obj

def _boundary_neighbors(edge_index, forget_nodes):
    """Neighbors of forget set (excluding the forget nodes)."""
    src, dst = edge_index
    fset = set(int(n) for n in forget_nodes.detach().cpu().tolist())
    nbrs = []
    for u, v in zip(src.detach().cpu().tolist(), dst.detach().cpu().tolist()):
        if u in fset and v not in fset: nbrs.append(v)
        if v in fset and u not in fset: nbrs.append(u)
    return list(sorted(set(nbrs)))

def prepare_pre_unlearning(dataset: str = "physics", base_epochs: int = 100,
                           forget_ratio: float = 0.05, backbone: str = "gcn"):
    dataset_obj = _load_dataset(dataset)
    data = dataset_obj[0]
    data = split_data(data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = get_model(backbone, data.num_node_features, 64, dataset_obj.num_classes).to(device)
    train_loader = NeighborLoader(data, num_neighbors=[5,10], batch_size=64, input_nodes=data.train_mask)
    val_loader = NeighborLoader(data, num_neighbors=[5,10], batch_size=64, input_nodes=data.val_mask)
    model.fit(data, base_epochs, train_loader, val_loader)  # p.4 Impl. Details. :contentReference[oaicite:9]{index=9}

    k = max(1, int(forget_ratio * data.num_nodes))
    forget_nodes = torch.randperm(data.num_nodes, device=device)[:k]
    _ensure_no_grad_features_(data)
    return model, data, forget_nodes

# ──────────────────────────────────────────────────────────────
# Unlearning strategies (return 5‑tuple: model, data_post, time, mapped, post_proxy_targets)
# ──────────────────────────────────────────────────────────────
def retrain_from_scratch(model, data, forget_nodes, backbone: Optional[str] = None):
    print("\n[During Unlearning] Full model reinit + retraining...")
    device = data.edge_index.device
    remaining_nodes = torch.tensor([n for n in range(data.num_nodes) if n not in forget_nodes], device=device)

    node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(remaining_nodes)}
    mapped_forget = torch.tensor([node_map[n.item()] for n in forget_nodes if n.item() in node_map],
                                 dtype=torch.long, device=device)

    # boundary neighbors in pre graph, then map into post indices
    pre_boundary = _boundary_neighbors(data.edge_index, forget_nodes)
    post_proxy_targets = torch.tensor([node_map[u] for u in pre_boundary if u in node_map],
                                      dtype=torch.long, device=device)

    sub_edge_index, _ = subgraph(remaining_nodes, data.edge_index, relabel_nodes=True)
    x_sub = data.x[remaining_nodes].detach().clone()
    sub_data = data.__class__(x=x_sub, edge_index=sub_edge_index, y=data.y[remaining_nodes]).to(device)
    sub_data = split_data(sub_data); _ensure_no_grad_features_(sub_data)

    bb = (backbone or infer_backbone(model))
    out_dim = int(torch.max(sub_data.y).item() + 1)
    new_model = get_model(bb, sub_data.num_node_features, 64, out_dim).to(device)

    train_loader = NeighborLoader(sub_data, num_neighbors=[5,10], batch_size=64, input_nodes=sub_data.train_mask)
    val_loader = NeighborLoader(sub_data, num_neighbors=[5,10], batch_size=64, input_nodes=sub_data.val_mask)
    start = time.time(); new_model.fit(sub_data, 100, train_loader, val_loader); end = time.time()
    return new_model, sub_data, end - start, mapped_forget, post_proxy_targets

def graph_delete(model, data, forget_nodes, backbone: Optional[str] = None):
    print("\n[During Unlearning] GNNDelete‑style subgraph removal + finetune...")
    device = data.edge_index.device
    remaining_nodes = torch.tensor([n for n in range(data.num_nodes) if n not in forget_nodes],
                                   dtype=torch.long, device=device)
    node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(remaining_nodes)}
    mapped = torch.tensor([node_map[n.item()] for n in forget_nodes if n.item() in node_map],
                          dtype=torch.long, device=device)
    pre_boundary = _boundary_neighbors(data.edge_index, forget_nodes)
    post_proxy_targets = torch.tensor([node_map[u] for u in pre_boundary if u in node_map],
                                      dtype=torch.long, device=device)

    sub_edge_index, _ = subgraph(remaining_nodes, data.edge_index, relabel_nodes=True)
    x_sub = data.x[remaining_nodes].detach().clone()
    sub_data = data.__class__(x=x_sub, edge_index=sub_edge_index, y=data.y[remaining_nodes]).to(device)
    sub_data = split_data(sub_data); _ensure_no_grad_features_(sub_data)

    train_loader = NeighborLoader(sub_data, num_neighbors=[5,10], batch_size=64, input_nodes=sub_data.train_mask)
    val_loader = NeighborLoader(sub_data, num_neighbors=[5,10], batch_size=64, input_nodes=sub_data.val_mask)
    start = time.time(); model.fit(sub_data, epochs=50, train_loader=train_loader, val_loader=val_loader); end = time.time()
    return model, sub_data, end - start, mapped, post_proxy_targets

def graph_remover(model, data, forget_nodes, backbone: Optional[str] = None):
    print("\n[During Unlearning] GraphRemover: zeroing features/embeddings...")
    start = time.time()
    with torch.no_grad():
        data.x[forget_nodes] = 0.0
    end = time.time()
    _ensure_no_grad_features_(data)
    post_proxy_targets = forget_nodes  # nodes exist
    return model, data, end - start, forget_nodes, post_proxy_targets

def idea_method(model, data, forget_nodes, backbone: Optional[str] = None):
    print("\n[During Unlearning] IDEA‑style re‑initialization and training...")
    device = data.edge_index.device
    bb = (backbone or infer_backbone(model))
    out_dim = int(data.y.max().item() + 1)
    new_model = get_model(bb, data.num_node_features, 64, out_dim).to(device)

    _ensure_no_grad_features_(data)
    train_loader = NeighborLoader(data, num_neighbors=[5,10], batch_size=64, input_nodes=data.train_mask)
    val_loader = NeighborLoader(data, num_neighbors=[5,10], batch_size=64, input_nodes=data.val_mask)
    start = time.time(); new_model.fit(data, epochs=50, train_loader=train_loader, val_loader=val_loader); end = time.time()
    post_proxy_targets = forget_nodes  # nodes remain
    return new_model, data, end - start, forget_nodes, post_proxy_targets

def graph_editor(model, data, forget_nodes, backbone: Optional[str] = None):
    print("\n[During Unlearning] GraphEditor‑inspired feature/edge corruption + finetune...")
    device = data.edge_index.device
    data = data.clone().to(device)

    # Partial edge drop to avoid isolating the forget set completely
    drop_ratio = 0.6  # tune if needed; 0.6 keeps 40% of incident edges
    fn = forget_nodes.to(data.edge_index.device)
    inc = (
        (data.edge_index[0].unsqueeze(0) == fn.view(-1,1)).any(0) |
        (data.edge_index[1].unsqueeze(0) == fn.view(-1,1)).any(0)
    ).nonzero(as_tuple=False).view(-1)
    keep_mask = torch.ones(data.edge_index.size(1), dtype=torch.bool, device=data.edge_index.device)
    if inc.numel() > 0:
        k = max(0, int((1.0 - drop_ratio) * inc.numel()))
        # randomly keep a subset of incident edges (others dropped)
        perm = torch.randperm(inc.numel(), device=inc.device)
        keep_idx = inc[perm[:k]]
        keep_mask[inc] = False
        keep_mask[keep_idx] = True
    data.edge_index = data.edge_index[:, keep_mask]

    # slight feature corruption
    data.x[forget_nodes] = data.x[forget_nodes] + 0.01 * torch.randn_like(data.x[forget_nodes])
    _ensure_no_grad_features_(data)

    train_loader = NeighborLoader(data, num_neighbors=[5,10], batch_size=64, input_nodes=data.train_mask)
    val_loader = NeighborLoader(data, num_neighbors=[5,10], batch_size=64, input_nodes=data.val_mask)
    model.fit(data, 50, train_loader=train_loader, val_loader=val_loader)
    post_proxy_targets = forget_nodes  # nodes remain
    end = time.time()
    return model, data, end - time.time() + end, forget_nodes, post_proxy_targets  # maintain signature

def node_unlearning(model, data, forget_nodes, backbone: Optional[str] = None):
    print("\n[During Unlearning] NodeUnlearning: localized subgraph retraining...")
    from torch_geometric.utils import degree
    device = data.edge_index.device

    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    connected_to_remove = [node.item() for node in forget_nodes if deg[node] > 0]
    if not connected_to_remove:
        print("No connected nodes found to unlearn. Skipping...")
        post_proxy_targets = forget_nodes
        return model, data, 0.0, forget_nodes, post_proxy_targets

    remaining_nodes = torch.tensor(
        [n for n in range(data.num_nodes) if n not in connected_to_remove],
        dtype=torch.long, device=device
    )
    node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(remaining_nodes)}
    mapped = torch.tensor([node_map[n] for n in forget_nodes if n.item() in node_map],
                          dtype=torch.long, device=device)
    pre_boundary = _boundary_neighbors(data.edge_index, forget_nodes)
    post_proxy_targets = torch.tensor([node_map[u] for u in pre_boundary if u in node_map],
                                      dtype=torch.long, device=device)

    sub_edge_index, _ = subgraph(remaining_nodes, data.edge_index, relabel_nodes=True)
    x_sub = data.x[remaining_nodes].detach().clone()
    sub_data = data.__class__(x=x_sub, edge_index=sub_edge_index, y=data.y[remaining_nodes]).to(device)
    sub_data = split_data(sub_data); _ensure_no_grad_features_(sub_data)

    train_loader = NeighborLoader(sub_data, num_neighbors=[5,10], batch_size=64, input_nodes=sub_data.train_mask)
    val_loader = NeighborLoader(sub_data, num_neighbors=[5,10], batch_size=64, input_nodes=sub_data.val_mask)

    start = time.time(); model.fit(sub_data, epochs=50, train_loader=train_loader, val_loader=val_loader); end = time.time()
    return model, sub_data, end - start, mapped, post_proxy_targets

def perform_unlearning(strategy="idea", model=None, data=None, forget_nodes=None, backbone: Optional[str] = None):
    if strategy == "retrain":
        return retrain_from_scratch(model, data, forget_nodes, backbone)
    elif strategy == "grapheditor":
        return graph_editor(model, data, forget_nodes, backbone)
    elif strategy == "gnndelete":
        return graph_delete(model, data, forget_nodes, backbone)
    elif strategy == "idea":
        return idea_method(model, data, forget_nodes, backbone)
    elif strategy == "nodeunlearning":
        return node_unlearning(model, data, forget_nodes, backbone)
    elif strategy == "graphremover":
        return graph_remover(model, data, forget_nodes, backbone)
    else:
        raise ValueError(f"Unknown unlearning strategy: {strategy}")
