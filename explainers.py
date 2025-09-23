# explainers.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
import torch
from torch_geometric.utils import k_hop_subgraph

def extract_paths_from_tree(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    paths = []
    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]; threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], path + [f"{name} <= {threshold:.4f}"])
            recurse(tree_.children_right[node], path + [f"{name} > {threshold:.4f}"])
        else:
            value = tree_.value[node][0]; class_id = value.argmax()
            rule = " ∧ ".join(path) + f" → class {class_id}"; paths.append(rule)
    recurse(0, []); return paths

def run_graphchef(stage="pre", model=None, data=None):
    print(f"[{stage}] Extracting rules using GraphChef ...")
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred_y = out.argmax(dim=1).cpu().numpy()
    X = data.x.detach().cpu().numpy()
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, pred_y)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    rules = extract_paths_from_tree(clf, feature_names)
    return {"rules": rules, "clf": clf}

def run_proxy_graph_generator(stage="pre", model=None, data=None, target_nodes=None, hops: int = 2):
    print(f"[{stage}] Generating Proxy Graph via {hops}-hop ego graph(s) …")
    if target_nodes is None:
        target_nodes = []
    if isinstance(target_nodes, torch.Tensor):
        target_nodes = [int(n.item()) for n in target_nodes]
    else:
        target_nodes = [int(n) for n in target_nodes]
    if len(target_nodes) == 0:
        return {"graph": [], "nodes": []}

    max_idx = int(data.num_nodes) - 1
    valid_targets = [n for n in target_nodes if 0 <= n <= max_idx]
    if not valid_targets:
        print(f"[{stage}] No valid target nodes inside graph – returning empty proxy.")
        return {"graph": [], "nodes": []}

    edge_index_cpu = data.edge_index.detach().cpu()
    edge_set, node_set = set(), set()
    for tgt in valid_targets:
        try:
            subset, edge_index, _, _ = k_hop_subgraph(tgt, hops, edge_index_cpu, relabel_nodes=False)
            node_set.update([int(i) for i in subset.tolist()])
            for e in edge_index.t().tolist():
                edge_set.add(tuple(sorted(e)))
        except Exception as e:
            print(f"[{stage}] Warning: skipping target {tgt} in proxy graph gen ({e})")
            continue
    edges = [list(e) for e in edge_set]
    return {"graph": edges, "nodes": list(node_set)}
