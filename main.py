# main.py
import os
import random
import pandas as pd
import numpy as np
from mi import run_mi_attacks, run_mi_forgetset_vs_shadow  # add import
from explainers import run_graphchef, run_proxy_graph_generator
from unlearning import prepare_pre_unlearning, perform_unlearning
from metrics import calculate_residual_attribution, evaluate_metrics
from visualization import (
    plot_rule_diff, plot_graph_diff, plot_ra_per_node,
    plot_explanation_shift, plot_rule_counts,
    visualize_graphchef_rules, compare_graphchef_trees,
    plot_attribution_histogram, plot_attribution_scatter,
    plot_batch_comparison
)
from mi import run_mi_attacks  # membership inference

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def normalize_rule(rule):
    return rule.strip().replace(" ", "").lower()

def get_removed_and_modified_rules(chef_pre_rules, chef_post_rules):
    removed_rules = []
    modified_rules = []
    pre_rule_set = set(chef_pre_rules)
    post_rule_set = set(chef_post_rules)
    removed_rules = pre_rule_set - post_rule_set
    for rule_pre in chef_pre_rules:
        if rule_pre not in chef_post_rules:
            for rule_post in chef_post_rules:
                if rule_pre.split('→')[1] == rule_post.split('→')[1]:
                    modified_rules.append(rule_pre)
    return removed_rules, modified_rules

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────
# Single end-to-end (dataset/strategy/backbone)
# ──────────────────────────────────────────────────────────────
def run_single(dataset: str, strategy: str, seed: int = 0, backbone: str = "gcn"):
    set_seed(seed)

    # Stage 1: pre-unlearning
    model, data, forget_nodes = prepare_pre_unlearning(dataset=dataset, backbone=backbone)
    ra_pre, grad_pre = calculate_residual_attribution(model, data, forget_nodes)
    chef_pre = run_graphchef('pre', model, data)
    proxy_pre = run_proxy_graph_generator('pre', model, data, forget_nodes)

    mi_pre = run_mi_attacks(model, data, forget_nodes)
    mi_pre_F = run_mi_forgetset_vs_shadow(model, data, forget_nodes)  # NEW

    # Stage 2: unlearning (returns mapped forgotten ids *and* post proxy targets)
    model, data_post, unlearn_time, mapped, post_proxy_targets = perform_unlearning(
        strategy=strategy, model=model, data=data, forget_nodes=forget_nodes, backbone=backbone
    )

    # Stage 3: post-unlearning
    ra_post, grad_post = calculate_residual_attribution(model, data_post, mapped)
    chef_post = run_graphchef('post', model, data_post)
    proxy_post = run_proxy_graph_generator('post', model, data_post, post_proxy_targets)
    mi_post = run_mi_attacks(model, data_post, mapped)
    mi_post_F = run_mi_forgetset_vs_shadow(model, data_post, mapped)  # NEW

    # Core metrics (strictly per paper)
    results = evaluate_metrics(
        chef_pre, chef_post, proxy_pre, proxy_post,
        ra_pre, ra_post, grad_pre, grad_post,
        model, data_post, unlearn_time, forget_nodes
    )

    # Rule diagnostics (console only)
    removed_rules, modified_rules = get_removed_and_modified_rules(
        chef_pre["rules"], chef_post["rules"]
    )
    print(f"Rules before unlearning:")
    print("\n".join([normalize_rule(r) for r in chef_pre["rules"]]))
    print(f"\nRules after unlearning:")
    print("\n".join([normalize_rule(r) for r in chef_post["rules"]]))
    print(f"\nRemoved Rules: {len(removed_rules)}")
    print("\n".join(removed_rules))
    print(f"\nModified Rules: {len(modified_rules)}")
    print("\n".join(modified_rules))

    # Plots (optional, keep your paper figures)
    #plot_ra_per_node(forget_nodes, grad_pre, grad_post)
    #plot_explanation_shift(grad_pre, grad_post)
    #plot_rule_diff(chef_pre["rules"], chef_post["rules"])
    #plot_graph_diff(proxy_pre["graph"], proxy_post["graph"])
    #plot_rule_counts(chef_pre["rules"], chef_post["rules"])
    #plot_attribution_histogram(grad_pre, grad_post)
    #plot_attribution_scatter(grad_pre, grad_post)

    # Attach MI block for the CSV (kept separate from paper CSV)
    def safe_float(x, n=4):
        try: return round(float(x), n)
        except Exception: return x

    results.update({
        "Dataset": dataset,
        "Method": strategy.capitalize(),
        "Backbone": backbone.lower(),
        "Seed": seed,
        "MI Conf AUC (Pre)": safe_float(mi_pre["MI/Confidence AUC"]),
        "MI Conf AUC (Post)": safe_float(mi_post["MI/Confidence AUC"]),
        "MI Conf AUC Δ": safe_float(mi_pre["MI/Confidence AUC"] - mi_post["MI/Confidence AUC"]),
        "MI Loss AUC (Pre)": safe_float(mi_pre["MI/Loss AUC"]),
        "MI Loss AUC (Post)": safe_float(mi_post["MI/Loss AUC"]),
        "MI Loss AUC Δ": safe_float(mi_pre["MI/Loss AUC"] - mi_post["MI/Loss AUC"]),
        "MI Conf Adv (Pre)": safe_float(mi_pre["MI/Confidence Adv"]),
        "MI Conf Adv (Post)": safe_float(mi_post["MI/Confidence Adv"]),
        "MI Loss Adv (Pre)": safe_float(mi_pre["MI/Loss Adv"]),
        "MI Loss Adv (Post)": safe_float(mi_post["MI/Loss Adv"]),
        "MI-F Conf AUC (Pre)": mi_pre_F["MI-F/Confidence AUC"],
        "MI-F Conf AUC (Post)": mi_post_F["MI-F/Confidence AUC"],
        "MI-F Loss AUC (Pre)": mi_pre_F["MI-F/Loss AUC"],
        "MI-F Loss AUC (Post)": mi_post_F["MI-F/Loss AUC"],
        "MI-F Conf Adv (Pre)": mi_pre_F["MI-F/Confidence Adv"],
        "MI-F Conf Adv (Post)": mi_post_F["MI-F/Confidence Adv"],
        "MI-F Loss Adv (Pre)": mi_pre_F["MI-F/Loss Adv"],
        "MI-F Loss Adv (Post)": mi_post_F["MI-F/Loss Adv"],
        "Mean Conf on Forget (Pre)": safe_float(mi_pre["MI/MeanConf@Forget"]),
        "Mean Conf on Forget (Post)": safe_float(mi_post["MI/MeanConf@Forget"]),
        "Mean −Loss on Forget (Pre)": safe_float(mi_pre["MI/MeanNegLoss@Forget"]),
        "Mean −Loss on Forget (Post)": safe_float(mi_post["MI/MeanNegLoss@Forget"]),
    })

    # Convert 'RA Δ' from "x%" → float for aggregation/plots
    if isinstance(results.get('RA Δ'), str) and results['RA Δ'].endswith('%'):
        results['RA Δ'] = float(results['RA Δ'].replace('%',''))

    return results

def _aggregate_and_save(all_records, agg_path="batch_unlearning_results_agg.csv"):
    df = pd.DataFrame(all_records)
    df.to_csv("batch_unlearning_results.csv", index=False, encoding='utf-8-sig')
    print("\nPer‑run results saved to batch_unlearning_results.csv\n")

    meta_cols = ["Dataset", "Method", "Backbone"]
    num_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    grouped = df.groupby(meta_cols)[num_cols].agg(['mean', 'std'])
    grouped.columns = [f"{col} ({stat})" for col, stat in grouped.columns]
    grouped = grouped.reset_index()
    grouped.to_csv(agg_path, index=False, encoding='utf-8-sig')
    print(f"Aggregated mean±std saved to {agg_path}\n")

    # Paper-only CSV (mirror manuscript): RA Δ, HS, ESD, GED Δ, Rules Removed, Time
    mean_df = df.groupby(meta_cols)[num_cols].mean().reset_index()
    paper_cols = ['Dataset','Method','Backbone','RA Δ','HS','ESD','GED Δ','Rules Removed','Unlearn Time (s)']
    paper_df = mean_df[[c for c in paper_cols if c in mean_df.columns]].copy()
    paper_df.to_csv('paper_metrics_only.csv', index=False, encoding='utf-8-sig')
    print("Paper-only metrics saved to paper_metrics_only.csv")

    # Per-backbone plots (mean values)
    for b in mean_df["Backbone"].unique():
        sub = mean_df[mean_df["Backbone"] == b].copy()
        plot_batch_comparison(sub, save_path=f"batch_comparison_{b}.png")
    return df, grouped, mean_df

# ──────────────────────────────────────────────────────────────
def _parse_backbones_env():
    spec = os.environ.get("BACKBONE", "") or os.environ.get("BACKBONES", "")
    if not spec:
        return ["gcn", "graphsage", "gat"]
    items = [s.strip().lower() for s in spec.split(",") if s.strip()]
    valid = {"gcn", "graphsage", "gat"}
    items = [b for b in items if b in valid]
    return items or ["gcn", "graphsage", "gat"]

def run_all(repeats=5, backbones=None):
    if backbones is None:
        backbones = ["gcn", "graphsage", "gat"]
    datasets = ['cora','citeseer','pubmed','CS','Physics']
    
    strategies = ['retrain','grapheditor','gnndelete','idea']
    
    all_records = []
    for backbone in backbones:
        for ds in datasets:
            for strat in strategies:
                for r in range(repeats):
                    print(f"=== Running {strat} on {ds} | {backbone} | run {r+1}/{repeats} ===")
                    rec = run_single(ds, strat, seed=1000 + r, backbone=backbone)
                    all_records.append(rec)
    _aggregate_and_save(all_records)

if __name__ == "__main__":
    repeats = int(os.environ.get("REPEATS", 3))
    backbones = _parse_backbones_env()  # default
    run_all(repeats=repeats, backbones=backbones)

