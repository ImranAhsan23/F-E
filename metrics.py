# metrics.py
import torch
import torch.nn.functional as F
from numpy import array
from numpy.linalg import norm

def evaluate_metrics(
    chef_pre, chef_post, proxy_pre, proxy_post,
    ra_pre, ra_post, grad_pre, grad_post,
    model, data, unlearn_time, forget_nodes,
):
    # Rules (RS)
    rules_pre = {r.strip().replace(" ","").lower() for r in chef_pre["rules"]}
    rules_post = {r.strip().replace(" ","").lower() for r in chef_post["rules"]}
    rules_removed = len(rules_pre - rules_post)

    # GED Δ (edge symmetric difference on proxy graphs)
    edges_pre = {tuple(sorted(e)) for e in proxy_pre["graph"]}
    edges_post = {tuple(sorted(e)) for e in proxy_post["graph"]}
    ged_val = len(edges_pre.symmetric_difference(edges_post))

    # HS = mean absolute per-node change
    g0 = array(grad_pre); g1 = array(grad_post)
    m = min(len(g0), len(g1))
    hs = float(abs(g0[:m] - g1[:m]).mean()) if m > 0 else 0.0
    hs = round(hs, 4)

    # ESD = mean absolute change on forgotten nodes only
    forget = [int(n.item()) if hasattr(n, "item") else int(n) for n in forget_nodes]
    esd_terms = []
    for v in forget:
        a_pre = grad_pre[v] if v < len(grad_pre) else 0.0
        a_post = grad_post[v] if v < len(grad_post) else 0.0
        esd_terms.append(abs(a_pre - a_post))
    esd = round(float(sum(esd_terms) / max(len(esd_terms), 1)), 4)

    return {
        "RA (Pre)": f"{ra_pre}%",
        "RA (Post)": f"{ra_post}%",
        "RA Δ": f"{round(ra_pre - ra_post, 2)}%",
        "HS": hs,
        "ESD": esd,
        "GED (Pre)": len(edges_pre),
        "GED (Post)": len(edges_post),
        "GED Δ": ged_val,
        "Rules (Pre)": len(rules_pre),
        "Rules (Post)": len(rules_post),
        "Rules Removed": rules_removed,
        "Unlearn Time (s)": round(unlearn_time, 2),
        "Model": type(model).__name__,
        "Dataset": "physics",
    }

def calculate_residual_attribution(model, data, nodes):
    """
    RA uses cross-entropy as in Eq. (1) (p.2).
    Returns: (RA%, grad_vector)
    """
    # fresh, grad-enabled copy of features
    x_clone = data.x.detach().clone().requires_grad_(True)
    x_old = data.x
    data.x = x_clone
    try:
        model.eval()
        out = model(data.x, data.edge_index)  # log-softmax
        if hasattr(data, "train_mask") and data.train_mask is not None and data.train_mask.any():
            mask = data.train_mask
        else:
            mask = torch.arange(data.num_nodes, device=out.device)
        loss = F.nll_loss(out[mask], data.y[mask])
        if data.x.grad is not None:
            data.x.grad.zero_()
        loss.backward()
        grad = data.x.grad.abs().sum(dim=1).detach().cpu()  # [N]
        total = max(float(grad.sum().item()), 1e-12)
        idx = nodes.cpu() if hasattr(nodes, "cpu") else nodes
        forget_score = float(grad[idx].sum().item()) if len(idx) > 0 else 0.0
        ra = round((forget_score / total) * 100, 2)
        return ra, grad.tolist()
    finally:
        data.x = data.x.detach()
        data.x.requires_grad_(False)
