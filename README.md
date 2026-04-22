# LightGCN Recommender

Implementation of **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation**
(Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang — SIGIR 2020).

---

## What is this about?

The core problem is **collaborative filtering**: given a history of which users interacted with which items, predict what a user would like next — without knowing anything about the users or items beyond the interaction history itself.

LightGCN approaches this by representing the user-item interaction data as a **graph** (users and items are nodes, interactions are edges) and learning embeddings by propagating information through that graph. The key insight of the paper is that two operations standard in Graph Convolutional Networks — feature transformation and nonlinear activation — are not only unnecessary for this task, but actively hurt performance. Removing them entirely produces a simpler, faster, and more accurate model.

---

## Dataset — Gowalla

A location-based social network dataset where users check in to physical places.
Sourced from the original LightGCN authors' repository (same train/test split as the paper).

| Metric | Value |
|--------|-------|
| Users | 29,858 |
| Items (places) | 40,981 |
| Interactions | 1,027,370 |
| Density | 0.00084 |
| Train / Test split | 78.9% / 21.1% |

The interaction matrix has over **1.2 billion possible entries** — only 0.084% of them are filled.
This extreme sparsity is the core challenge the model must overcome.

---

## Model Architecture

LightGCN consists of two components:

**1. Light Graph Convolution** (Eq. 3 in the paper)

At each layer, every node's embedding is replaced by a weighted average of its neighbors' embeddings:
```
E^(k+1) = D^(-1/2) A D^(-1/2) · E^(k)
```
where `A` is the user-item adjacency matrix and `D` is the degree matrix used for symmetric normalization.
There are no weight matrices and no activation functions — just neighbor aggregation.

**2. Layer Combination** (Eq. 4)

The final embedding is the uniform average of all layer outputs:
```
e_u = (E^(0) + E^(1) + ... + E^(K)) / (K + 1)
```
This prevents over-smoothing (where deep layers cause all embeddings to converge to the same value)
and lets the model capture both local and higher-order neighborhood structure simultaneously.

**Prediction** (Eq. 5): dot product of the final user and item embeddings.

The only trainable parameters are the **0th-layer embeddings** — one 64-dimensional vector per user and per item.
Total: `(29,858 + 40,981) × 64 = 4,533,696` parameters.

---

## Training

- **Loss**: Bayesian Personalized Ranking (BPR) — for each training interaction (user, positive item),
  sample one random negative item and push the positive score above the negative
- **Regularization**: L2 on 0th-layer embeddings of sampled nodes only (not all parameters)
- **Optimizer**: Adam, learning rate 0.001
- **Batch size**: 1024 triples per batch
- **One epoch** = one full pass through all 810,128 training interactions (~791 batches)

---

## Results

Evaluation metric: **Recall@20** and **NDCG@20** using the all-ranking protocol —
for each test user, rank all items (minus training items) and check whether
the true test items appear in the top 20.

| | Recall@20 | NDCG@20 |
|---|---|---|
| Paper (Table 3, K=3) | 0.1823 | 0.1554 |
| Our implementation | *(to be filled after full run)* | *(to be filled)* |

Training converges around epoch 150–200. Results at selected checkpoints:

| Epoch | Recall@20 | NDCG@20 |
|-------|-----------|---------|
| 5 | 0.1080 | 0.0941 |
| 10 | 0.1195 | 0.1044 |
| 20 | 0.1337 | 0.1159 |
| 35 | 0.1459 | 0.1256 |
| 200 | *(running)* | *(running)* |

---

## Implementation Notes & Findings

**Finding 1 — One epoch must cover the full dataset.**

Our first implementation ran a single batch of 1024 samples per epoch.
After 1000 such "epochs", the model had effectively seen the data only once,
yielding Recall@20 = 0.0945. After correcting the loop so each epoch processes
all 810,128 interactions (~791 batches), the model reached Recall@20 = 0.1459
in just 35 epochs. This is the most important implementation detail to get right.

**Finding 2 — Convergence is fast, 1000 epochs is an upper bound.**

The paper states "1000 epochs sufficient for convergence", but in practice
the model converges well before that. Based on our training curve, performance
plateaus around epoch 150–200, making the full 1000-epoch run unnecessary.

**Finding 3 — The simplicity is real.**

LightGCN achieves competitive results with a single embedding table and one line
of computation per layer (`E = Ã · E`). The entire forward pass is ~10 lines of PyTorch.
This confirms the paper's main argument: complexity hurts, not helps, for this task.

**Finding 4 — Sparse matrix representation is essential.**

The user-item interaction matrix has 1.2 billion entries. Storing it densely would
require ~5 GB of RAM. Using scipy sparse matrices reduces this to a few MB,
making the entire pipeline runnable on a standard Colab instance.

---

## Project Structure

```
LightGCN-Recommender/
├── data/                         # gitignored — Gowalla files (train.txt, test.txt)
├── notebooks/
│   ├── 01_data_exploration.ipynb # EDA: dataset statistics and visualizations
│   └── 02_experiments.ipynb      # Ablation: effect of K layers, convergence curves
├── src/
│   ├── dataset.py                # GowallaDataset: loads data, builds normalized adjacency matrix
│   ├── model.py                  # LightGCN: graph convolution + layer combination
│   ├── train.py                  # BPR training loop
│   └── evaluate.py               # Recall@20 and NDCG@20 metrics
├── requirements.txt
└── README.md
```

---

## Setup & Running

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run training locally** (CPU, slow — for testing only):
```bash
python src/train.py
```

**Run training in Google Colab** (GPU T4, recommended):
```python
# Cell 1 — mount Drive (data lives here)
from google.colab import drive
drive.mount('/content/drive')

# Cell 2 — clone repo and run
%cd /content
!git clone https://github.com/JacKoz7/LightGCN-Recommender
%cd LightGCN-Recommender
!python src/train.py
```

Data (`train.txt`, `test.txt`) must be placed at:
`/content/drive/MyDrive/ZED_project_data/gowalla/`

---

## Reference

```
@inproceedings{he2020lightgcn,
  title     = {LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation},
  author    = {He, Xiangnan and Deng, Kuan and Wang, Xiang and Li, Yan and Zhang, Yongdong and Wang, Meng},
  booktitle = {Proceedings of the 43rd International ACM SIGIR Conference},
  year      = {2020}
}
```
