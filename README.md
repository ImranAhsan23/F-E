# Forget & Explain: Transparent Verification of GNN Unlearning

This repository contains the implementation for **explainability‑driven verification of GNN unlearning**.  
It supports **GCN, GraphSAGE, and GAT** backbones and includes **Membership‑Inference (MI)** checks so reviewers can see how our explainability metrics line up with a standard privacy test.

**Core idea.** We verify forgetting by comparing **pre‑ vs post‑unlearning explanations** (saliency, proxy graph structure, rule consistency). Key metrics:

- **RA Δ (Residual Attribution Δ)** — reduction in the fraction of saliency flowing through the forget set (Eq. 1 in the paper). Higher is better.
- **HS** — mean absolute change of per‑node attribution (higher is more change).
- **ESD** — mean absolute change on **only the forgotten nodes** (higher is more change in the forget region).
- **GED Δ** — symmetric difference in edges of the **proxy graphs** around the forget region (larger indicates greater structural edits).
- **RS (Rules Removed)** — number of GraphChef rules removed.

We also report **MI** (membership inference) before and after unlearning (confidence‑based and loss‑based AUCs). When forgetting succeeds, MI should trend toward chance and RA_post should be near zero.

---

## Environment

We recommend **Python 3.10** and a CUDA‑enabled PyTorch if you have a GPU (works on CPU as well).

**Conda (recommended on Windows):**
```bash
conda create -n gnn_unlearning python=3.10 -y
conda activate gnn_unlearning

# PyTorch (choose the CUDA build that matches your driver; CPU-only also works)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyG stack (version compatible with your torch; if needed see PyG install docs)
pip install torch-geometric
pip install pandas scikit-learn matplotlib networkx

## Running Program

python main.py
