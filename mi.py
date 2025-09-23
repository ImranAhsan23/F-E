# mi.py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def _per_node_stats(model, data):
    model.eval()
    with torch.no_grad():
        log_probs = model(data.x, data.edge_index)  # [N,C] log-softmax
    y = data.y.view(-1)
    true_logp = log_probs.gather(1, y.unsqueeze(1)).squeeze(1)
    probs_true = torch.exp(true_logp).cpu().numpy()
    neg_loss = true_logp.cpu().numpy()  # == -(CE)
    mem = data.train_mask.detach().cpu().numpy().astype(int)
    return probs_true, neg_loss, mem

def _attack_metrics(scores, mem):
    scores = np.asarray(scores); mem = np.asarray(mem)
    mask = np.isfinite(scores) & np.isfinite(mem)
    scores = scores[mask]; mem = mem[mask]
    if scores.size == 0 or len(np.unique(mem)) < 2:
        return {"AUC": float("nan"), "Acc": float("nan"), "Adv": float("nan"), "Thresh": float("nan")}
    auc = float(roc_auc_score(mem, scores))
    qs = np.linspace(0,1,101); thresholds = np.quantile(scores, qs)
    best_acc, best_thr = -1.0, thresholds[50]
    for t in thresholds:
        pred = (scores >= t).astype(int); acc = (pred == mem).mean()
        if acc > best_acc: best_acc, best_thr = acc, t
    pred = (scores >= best_thr).astype(int)
    pos = (mem == 1).sum(); neg = (mem == 0).sum()
    tpr = ((pred == 1) & (mem == 1)).sum() / max(pos, 1)
    fpr = ((pred == 1) & (mem == 0)).sum() / max(neg, 1)
    return {"AUC": auc, "Acc": best_acc, "Adv": float(tpr - fpr), "Thresh": float(best_thr)}

def run_mi_attacks(model, data, forget_nodes=None):
    probs_true, neg_loss, mem = _per_node_stats(model, data)
    conf_res = _attack_metrics(probs_true, mem)
    loss_res = _attack_metrics(neg_loss, mem)

    forget_idx = []
    if forget_nodes is not None:
        N = data.num_nodes
        if isinstance(forget_nodes, torch.Tensor):
            forget_nodes = forget_nodes.detach().cpu().numpy().tolist()
        forget_idx = [int(i) for i in forget_nodes if 0 <= int(i) < N]

    def _mean_on(idx, arr):
        if not idx: return float("nan")
        return float(np.mean([arr[i] for i in idx]))

    return {
        "MI/Confidence AUC": conf_res["AUC"],
        "MI/Loss AUC": loss_res["AUC"],
        "MI/Confidence Adv": conf_res["Adv"],
        "MI/Loss Adv": loss_res["Adv"],
        "MI/CalibThr/Conf": conf_res["Thresh"],
        "MI/CalibThr/Loss": loss_res["Thresh"],
        "MI/MeanConf@Forget": _mean_on(forget_idx, probs_true),
        "MI/MeanNegLoss@Forget": _mean_on(forget_idx, neg_loss),
    }

def run_mi_forgetset_vs_shadow(model, data, forget_nodes, shadow_size=None):
    """
    Targeted MI on the forget set: compare forgotten nodes against a matched
    set of *non-members* (data.train_mask==False) with the same class mix.
    Returns AUC/Adv using confidence and negative loss. If the forget set
    nodes do not exist in `data` (e.g., after retrain/GNNDelete), this will
    silently skip missing indices and use the ones that remain.
    """
    model.eval()
    with torch.no_grad():
        logp = model(data.x, data.edge_index)  # [N,C], log-softmax
    y = data.y.view(-1)
    true_logp = logp.gather(1, y.unsqueeze(1)).squeeze(1)
    conf = torch.exp(true_logp).cpu().numpy()
    neg_loss = true_logp.cpu().numpy()  # == -(CE)

    N = data.num_nodes
    F = []
    for n in (forget_nodes.tolist() if isinstance(forget_nodes, torch.Tensor) else list(forget_nodes)):
        if 0 <= int(n) < N:
            F.append(int(n))
    if len(F) == 0:
        return {
            "MI-F/Confidence AUC": float("nan"),
            "MI-F/Loss AUC": float("nan"),
            "MI-F/Confidence Adv": float("nan"),
            "MI-F/Loss Adv": float("nan"),
        }

    # candidates: non-members in this graph
    nonmem = (~data.train_mask).nonzero(as_tuple=False).view(-1).cpu().numpy().tolist()
    # stratify by label to reduce class-confound
    F_y = y[F].cpu().numpy()
    idx_shadow = []
    for cls in np.unique(F_y):
        need = int((F_y == cls).sum())
        pool = [i for i in nonmem if int(y[i].item()) == int(cls)]
        take = min(need, len(pool))
        if take > 0:
            idx_shadow += list(np.random.choice(pool, size=take, replace=False))
    if shadow_size:
        k = min(shadow_size, len(idx_shadow))
        idx_shadow = list(np.random.choice(idx_shadow, size=k, replace=False))
    # balance sizes
    k = min(len(F), len(idx_shadow))
    F = F[:k]; idx_shadow = idx_shadow[:k]
    if k == 0:
        return {
            "MI-F/Confidence AUC": float("nan"),
            "MI-F/Loss AUC": float("nan"),
            "MI-F/Confidence Adv": float("nan"),
            "MI-F/Loss Adv": float("nan"),
        }

    # labels: 1 for forget set, 0 for shadow non-members
    y_true = np.array([1]*k + [0]*k)
    s_conf = np.concatenate([conf[F], conf[idx_shadow]])
    s_nloss = np.concatenate([neg_loss[F], neg_loss[idx_shadow]])

    def _auc_adv(scores):
        try:
            auc = float(roc_auc_score(y_true, scores))
        except Exception:
            auc = float("nan")
        # advantage at the best accuracy threshold
        thr = np.quantile(scores, np.linspace(0,1,101))
        best_acc, best_t = -1.0, thr[50]
        for t in thr:
            pred = (scores >= t).astype(int)
            acc = (pred == y_true).mean()
            if acc > best_acc:
                best_acc, best_t = acc, t
        pred = (scores >= best_t).astype(int)
        pos = (y_true == 1).sum(); neg = (y_true == 0).sum()
        tpr = ((pred == 1) & (y_true == 1)).sum() / max(pos, 1)
        fpr = ((pred == 1) & (y_true == 0)).sum() / max(neg, 1)
        return auc, float(tpr - fpr)

    auc_c, adv_c = _auc_adv(s_conf)
    auc_l, adv_l = _auc_adv(s_nloss)

    return {
        "MI-F/Confidence AUC": round(auc_c, 4),
        "MI-F/Loss AUC": round(auc_l, 4),
        "MI-F/Confidence Adv": round(adv_c, 4),
        "MI-F/Loss Adv": round(adv_l, 4),
    }
