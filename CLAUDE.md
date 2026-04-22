# LightGCN Recommender — Project Context for Claude

## Project Goal

University graded project for the course "Zaawansowana eksploracja danych" (Advanced Data Exploration).
Implementation and experiments based on the paper:
**"LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"** (Xiangnan He et al., SIGIR 2020)
arXiv: https://arxiv.org/abs/2002.02126

The grade depends on:
- **50% implementation quality** — correctness, faithfulness to the paper, clean code
- **50% presentation quality** — clear explanation of the problem and results (20–25 min demo)

## Team

Two-person team. The user (Jacek) handles **all code implementation**.

## Coding Rules — MANDATORY

- All variable names, function names, class names, and code comments must be in **English**.
- Do NOT add unnecessary comments — only explain non-obvious logic (the WHY, not the WHAT).
- Do NOT over-engineer. Implement exactly what is needed for the paper's experiments.
- Do NOT add error handling for impossible scenarios. Trust the data pipeline.
- Keep code clean, readable, and consistent.

## Tech Stack

| Tool | Purpose |
|------|---------|
| VS Code (local) | Code editing, pushed to GitHub repo `ZED_LightGCN` |
| GitHub | Version control. Folders: `data/`, `notebooks/`, `src/` |
| Google Drive | Storage for large datasets (Gowalla) — NOT in git |
| Google Colab | Heavy training runs (GPU T4), mounts Drive, clones GitHub repo |
| PyTorch | Deep learning framework for LightGCN |
| Jupyter Notebooks | Exploration and experiment analysis |

## Dataset

**Gowalla** — location check-in dataset, sourced from the original LightGCN authors' repository.
- Stored on Google Drive (too large for GitHub)
- Format: `train.txt` and `test.txt`, each line = `user_id item_id1 item_id2 ...`
- `data/` folder is in `.gitignore`

## Project Structure

```
LightGCN-Recommender/
├── data/                    # gitignored — datasets live here (or on Drive)
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA of Gowalla dataset
│   └── 02_experiments.ipynb        # Ablation study, result comparison
├── src/
│   ├── dataset.py           # GowallaDataset class — loads data, builds graph
│   ├── model.py             # LightGCN model in PyTorch
│   ├── train.py             # Training loop with BPR loss
│   ├── evaluate.py          # Recall@20 and NDCG@20 metrics
│   └── main.py              # Entry point (CLI)
├── CLAUDE.md
└── .gitignore
```

## What Each File Does (plain language)

### `notebooks/01_data_exploration.ipynb` ✅ DONE
Answers the question: "what does our data look like before we touch the model?"
- Loads train.txt and test.txt
- Verifies our numbers match Table 2 from the paper (users, items, interactions, density)
- Shows how active users are and how popular items are (long-tail distributions)
- Visualizes how sparse the interaction matrix is — almost entirely empty
- Confirms the train/test split is clean (no test user is missing from training)

### `src/dataset.py` ✅ DONE
Answers the question: "how do we turn raw text files into something the model can compute with?"
- Reads train.txt and test.txt into Python dictionaries
- Builds the interaction matrix R (users × items, 1 = interaction exists)
- Builds the full graph adjacency matrix A by stacking R and R-transposed (Eq. 6 from paper)
- Normalizes A into Ã using D^(-1/2) A D^(-1/2) so popular nodes don't dominate (Eq. 7)
- Converts the result into a PyTorch sparse tensor ready for the model

### `src/model.py` ⬜ TODO
Answers the question: "what is the LightGCN model and how does it compute recommendations?"
- Defines the LightGCN class in PyTorch
- Holds one embedding vector per user and per item (the only learned parameters)
- At each layer: multiplies embeddings by Ã — each node absorbs its neighbors' embeddings
- After K layers: averages all layer outputs as the final embedding (Eq. 4 from paper)
- Predicts a score for (user, item) as the dot product of their final embeddings

### `src/train.py` ✅ DONE
Answers the question: "how do we teach the model what good recommendations look like?"
- Runs the training loop (1000 epochs)
- At each step: samples a (user, positive item, negative item) triple
- Computes BPR loss — tries to make the score for the positive item higher than the negative
- Updates embeddings using Adam optimizer with L2 regularization
- Evaluates on the test set every 20 epochs and prints Recall@20 and NDCG@20

### `src/evaluate.py` ✅ DONE
Answers the question: "how do we measure if our recommendations are actually good?"
- For each test user: scores all 40,981 items, removes the ones already seen in training
- Takes the top 20 highest-scored items as the recommendation list
- Recall@20: what fraction of the user's true test items appear in the top 20?
- NDCG@20: same idea, but items ranked higher in the list count more than items ranked lower

### `notebooks/02_experiments.ipynb` ⬜ TODO
Answers the question: "how do our results compare to the paper, and what affects performance?"
- Runs training with K = 1, 2, 3, 4 layers and plots Recall@20 vs number of layers
- Plots training loss and Recall@20 over epochs (convergence curves)
- Compares final numbers to Table 3 from the paper
- Shows that LightGCN with layer combination beats LightGCN with only the last layer

## Implementation Plan (Paper-faithful)

### Stage 1 — Data Exploration (`01_data_exploration.ipynb`)
- Load Gowalla train/test splits
- Stats: number of users, items, interactions, sparsity
- Distributions: interactions per user and per item (log-log plots)
- Visualizations in the style of course lab notebooks

### Stage 2 — Dataset class (`src/dataset.py`)
- Parse `train.txt` / `test.txt`
- Build user-item bipartite graph
- Compute the symmetric normalized adjacency matrix (as in the paper, Eq. 7-8)
- Return sparse tensors for use in PyTorch

### Stage 3 — LightGCN Model (`src/model.py`)
- Embedding layer for users and items (no feature transformation)
- K-layer graph convolution: simple weighted sum aggregation, NO weight matrices, NO non-linearities
- Final embedding = weighted sum of all layer outputs (alpha_k = 1/(K+1))
- Prediction: inner product of user and item final embeddings

### Stage 4 — Training (`src/train.py`)
- Optimizer: Adam
- Loss: BPR (Bayesian Personalized Ranking) + L2 regularization on embeddings
- Negative sampling: one random negative item per positive interaction
- Matches authors' training protocol

### Stage 5 — Evaluation (`src/evaluate.py`)
- Metrics: **Recall@20** and **NDCG@20** (exactly as in the paper)
- Protocol: for each test user, rank ALL items minus training items

### Stage 6 — Experiments (`02_experiments.ipynb`)
- Effect of number of layers K: {1, 2, 3, 4}
- Effect of embedding dimension: {16, 32, 64}
- Comparison: LightGCN vs MF baseline (K=0)
- Convergence plots (loss and metrics vs. epoch)
- Results table compared to Table 3 in the paper

## Key Paper Details to Stay Faithful To

- **No feature transformation**: LightGCN removes the weight matrices W from standard GCN
- **No non-linear activation**: no ReLU or sigmoid in propagation
- **Symmetric normalization**: A_tilde = D^(-1/2) * A * D^(-1/2) where A is the user-item interaction matrix
- **Layer combination**: final embedding is uniform average of all layer embeddings (including layer 0)
- **BPR loss** with L2 regularization only on the 0-th layer embeddings (not all layers)
- **Reported results in paper**: Recall@20=0.1327, NDCG@20=0.0760 on Gowalla

## Colab Setup Cells

```python
# Cell 1 — Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2 — Clone repo and navigate
%cd /content
!git clone https://github.com/JacKoz7/LightGCN-Recommender
%cd LightGCN-Recommender
```

## User Background

Jacek is a graduate student who is a **complete beginner** in:
- Deep Learning
- PyTorch
- Graph Neural Networks (GCN)
- Recommender systems

All explanations must be given in plain language ("na chłopski rozum" — in simple terms), step by step.
Assume no prior knowledge of these topics.
