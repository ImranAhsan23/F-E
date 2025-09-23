# visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.tree import plot_tree

def plot_rule_diff(pre_rules, post_rules):
    removed = set(pre_rules) - set(post_rules)
    plt.figure(figsize=(8,4))
    labels = ["Kept", "Removed"]
    sizes = [len(pre_rules) - len(removed), len(removed)]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title("GraphChef Rule Retention")
    plt.savefig("rule_diff.png"); plt.show()

def plot_rule_counts(rules_pre, rules_post):
    plt.figure(figsize=(4,4))
    plt.bar(["Pre-Unlearning","Post-Unlearning"], [len(rules_pre), len(rules_post)], color=["blue","orange"])
    plt.title("Number of Rules Before and After Unlearning")
    plt.ylabel("Rule Count")
    plt.savefig("rule_count_change.png"); plt.show()

def plot_graph_diff(graph_pre, graph_post):
    before = set(tuple(e) for e in graph_pre)
    after  = set(tuple(e) for e in graph_post)
    removed = before - after; kept = before & after
    plt.figure(figsize=(6,4))
    plt.bar(["Kept Edges","Removed Edges"], [len(kept), len(removed)], color=["green","red"])
    plt.title("Proxy Graph Edge Comparison")
    plt.savefig("graph_diff.png"); plt.show()

def plot_ra_per_node(forget_nodes, grad_pre, grad_post, save_path="ra_barplot.png"):
    nodes = [int(n.item()) if isinstance(n, torch.Tensor) else int(n) for n in forget_nodes]
    valid = [n for n in nodes if n < len(grad_pre) and n < len(grad_post)]
    ra_pre_vals = [grad_pre[n] for n in valid]; ra_post_vals = [grad_post[n] for n in valid]
    idx = list(range(len(valid))); width = 0.35
    plt.figure(figsize=(12,5))
    plt.bar([i - width/2 for i in idx], ra_pre_vals, width=width, label='Pre')
    plt.bar([i + width/2 for i in idx], ra_post_vals, width=width, label='Post')
    plt.xticks(idx, [str(n) for n in valid], rotation=90)
    plt.xlabel("Forgotten Node Index"); plt.ylabel("Attribution (Gradient Sum)")
    plt.title("Residual Attribution per Forgotten Node"); plt.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.show()

def plot_explanation_shift(grad_pre, grad_post, save_path="explanation_shift.png"):
    m = min(len(grad_pre), len(grad_post))
    x = list(range(m))
    plt.figure(figsize=(12,4))
    plt.plot(x, grad_pre[:m], label="Pre", alpha=0.7)
    plt.plot(x, grad_post[:m], label="Post", alpha=0.7)
    plt.title("Gradient Attribution Comparison (HS / ESD)")
    plt.xlabel("Node Index"); plt.ylabel("Attribution Score")
    plt.legend(); plt.tight_layout(); plt.savefig(save_path); plt.show()

def visualize_graphchef_rules(clf, feature_names, save_path="graphchef_rules.png"):
    plt.figure(figsize=(12,6))
    plot_tree(clf, feature_names=feature_names, filled=True, rounded=True, fontsize=8)
    plt.title("GraphChef Extracted Rules"); plt.tight_layout()
    plt.savefig(save_path); plt.show()

def compare_graphchef_trees(clf_pre, clf_post, feature_names, save_path="graphchef_tree_diff.png"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(20,8))
    plot_tree(clf_pre, feature_names=feature_names, filled=True, rounded=True, fontsize=8, ax=axes[0])
    axes[0].set_title("GraphChef Tree: Pre-Unlearning")
    plot_tree(clf_post, feature_names=feature_names, filled=True, rounded=True, fontsize=8, ax=axes[1])
    axes[1].set_title("GraphChef Tree: Post-Unlearning")
    plt.tight_layout(); plt.savefig(save_path); plt.show()

def plot_attribution_histogram(grad_pre, grad_post, bins=50, save_path="attribution_histogram.png"):
    plt.figure(figsize=(6,4))
    plt.hist(grad_pre, bins=bins, alpha=0.5, label='Pre')
    plt.hist(grad_post, bins=bins, alpha=0.5, label='Post')
    plt.title("Attribution Distribution: Pre vs Post")
    plt.xlabel("Attribution Score"); plt.ylabel("Number of Nodes")
    plt.legend(); plt.tight_layout(); plt.savefig(save_path); plt.show()

def plot_attribution_scatter(grad_pre, grad_post, save_path="attribution_scatter.png"):
    m = min(len(grad_pre), len(grad_post))
    x = grad_pre[:m]; y = grad_post[:m]
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=10, alpha=0.6)
    mx = max(max(x or [1]), max(y or [1])); plt.plot([0, mx], [0, mx], linestyle='--')
    plt.title("Pre vs Post Attribution Scatter"); plt.xlabel("Pre"); plt.ylabel("Post")
    plt.tight_layout(); plt.savefig(save_path); plt.show()

def plot_batch_comparison(df, metrics=['RA Δ','HS','ESD','GED Δ','Rules Removed'], save_path='batch_comparison.png'):
    datasets = df['Dataset'].unique(); methods = df['Method'].unique()
    n_ds = len(datasets); n_m = len(methods); x = range(n_ds)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4), sharey=False)
    if len(metrics) == 1: axes = [axes]
    for ax, metric in zip(axes, metrics):
        width = 0.8 / n_m
        for i, method in enumerate(methods):
            vals = []
            for ds in datasets:
                v = df[(df['Dataset']==ds)&(df['Method']==method)][metric].values
                vals.append(v[0] if len(v)>0 else 0)
            ax.bar([xi + i*width for xi in x], vals, width=width, label=method)
        ax.set_title(metric)
        ax.set_xticks([xi + width*(n_m-1)/2 for xi in x])
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.set_ylabel(metric)
    fig.legend(methods, loc='upper right')
    plt.tight_layout(); plt.savefig(save_path); plt.show()
