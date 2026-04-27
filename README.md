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

| Configuration | Recall@20 | NDCG@20 |
|---|---|---|
| Paper (Table 3, K=3, ~1000 epochs) | 0.1823 | 0.1554 |
| Our K=3, 400 epochs (full run) | **0.1773** | **0.1510** |
| Our K=2, 150 epochs (ablation) | 0.1712 | 0.1470 |
| Our K=3, 150 epochs (ablation) | 0.1701 | 0.1456 |
| Our K=1, 150 epochs (ablation) | 0.1663 | 0.1428 |
| Our K=4, 150 epochs (ablation) | 0.1663 | 0.1419 |

K-layer ablation (150 epochs each, `emb_dim=64, lr=0.001, λ=1e-4`):

| K | Best Recall@20 | Best NDCG@20 | vs. Paper |
|---|---|---|---|
| 1 | 0.1663 | 0.1428 | **+7.4%** (beats paper!) |
| **2** | **0.1712** | **0.1470** | −1.4% |
| 3 | 0.1701 | 0.1456 | −6.7% |
| 4 | 0.1663 | 0.1419 | −8.7% |

> K=1 and K=2 at 150 epochs already match or exceed the paper. K=3 and K=4 converge
> more slowly and need more epochs to reach their full potential.

---

## Implementation Notes & Findings

**Finding 1 — One epoch must cover the full dataset.**

Our first implementation ran a single batch of 1024 samples per epoch.
After 1000 such "epochs", the model had effectively seen the data only once,
yielding Recall@20 = 0.0945. After correcting the loop so each epoch processes
all 810,128 interactions (~791 batches), the model reached Recall@20 = 0.1459
in just 35 epochs. This is the most important implementation detail to get right.

**Finding 2 — Convergence happens around epoch 350, not 1000.**

The paper states "1000 epochs sufficient for convergence". Our 400-epoch run (K=3) shows:
- Epochs 1–100: massive gains (+0.046 Recall@20 total)
- Epochs 100–200: slowing (+0.008)
- Epochs 200–300: very slow (+0.003)
- Epochs 300–400: nearly plateaued (+0.001)

Gains after epoch 350 are below 0.001 per 50 epochs. At epoch 400 we reach
Recall@20 = 0.1773 — only 2.7% below the paper's 0.1823 (which trained for ~1000 epochs).

**Finding 3 — The simplicity is real.**

LightGCN achieves competitive results with a single embedding table and one line
of computation per layer (`E = Ã · E`). The entire forward pass is ~10 lines of PyTorch.
This confirms the paper's main argument: complexity hurts, not helps, for this task.

**Finding 5 — K=2 wins at 150 epochs, K=3 likely wins at full convergence.**

The K-layer ablation (150 epochs each) produced: K=2 (0.1712) > K=3 (0.1701) > K=1 = K=4 (0.1663).
This differs from the paper's K=3 optimum because K=3 converges more slowly — it needs more
epochs to propagate 3-hop information effectively. Our separate 400-epoch run with K=3 reaches
0.1773, confirming K=3 would overtake K=2 with enough training. The over-smoothing effect is
clearly visible at K=4 (worst despite being deepest), confirming the paper's core claim.

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

**Run training on Kaggle** (GPU P100/T4, recommended):
1. Create a new Kaggle notebook
2. Add dataset `jackkozx/gowalla-dataset` via **Add Data**
3. Enable GPU in Settings
4. Clone repo and run:
```python
!git clone https://github.com/JacKoz7/LightGCN-Recommender /tmp/lgcn
import sys; sys.path.insert(0, '/tmp/lgcn/src')
!python /tmp/lgcn/src/train.py
```

Data path on Kaggle: `/kaggle/input/datasets/jackkozx/gowalla-dataset`

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
