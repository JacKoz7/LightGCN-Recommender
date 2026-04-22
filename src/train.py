import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import GowallaDataset
from model import LightGCN
from evaluate import recall_and_ndcg_at_k


def _sample_batch(train_user_items, n_items, batch_size):
    """Sample (user, positive_item, negative_item) triples for BPR training."""
    all_users = list(train_user_items.keys())
    users, pos_items, neg_items = [], [], []

    while len(users) < batch_size:
        user = random.choice(all_users)
        pos  = random.choice(train_user_items[user])
        neg  = random.randint(0, n_items - 1)
        while neg in train_user_items[user]:
            neg = random.randint(0, n_items - 1)
        users.append(user)
        pos_items.append(pos)
        neg_items.append(neg)

    return (torch.LongTensor(users),
            torch.LongTensor(pos_items),
            torch.LongTensor(neg_items))


def _bpr_loss(model, users, pos_items, neg_items, reg_lambda):
    """BPR loss + L2 regularization on 0th-layer embeddings (Eq. 15 from paper)."""
    user_emb, item_emb = model()

    u = user_emb[users]
    i = item_emb[pos_items]
    j = item_emb[neg_items]

    loss = -torch.log(torch.sigmoid((u * i).sum(-1) - (u * j).sum(-1))).mean()

    # regularize only the 0th-layer embeddings of nodes in this batch
    e0 = model.embedding.weight
    reg = (e0[users].norm(2).pow(2)
           + e0[model.n_users + pos_items].norm(2).pow(2)
           + e0[model.n_users + neg_items].norm(2).pow(2)) / len(users)

    return loss + reg_lambda * reg


def train(data_path, emb_dim=64, n_layers=3, lr=0.001, reg_lambda=1e-4,
          batch_size=1024, n_epochs=1000, eval_every=20, device="cpu"):

    print(f"Loading dataset from {data_path} ...")
    dataset  = GowallaDataset(data_path)
    norm_adj = dataset.norm_adj.to(device)

    # number of batches per epoch = one full pass through training interactions
    n_batches = max(1, len(dataset.train_pairs) // batch_size)

    model = LightGCN(
        n_users  = dataset.n_users,
        n_items  = dataset.n_items,
        emb_dim  = emb_dim,
        n_layers = n_layers,
        norm_adj = norm_adj,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training LightGCN — layers={n_layers}, dim={emb_dim}, "
          f"lr={lr}, λ={reg_lambda}, device={device}")
    print(f"Batches per epoch: {n_batches}  "
          f"({len(dataset.train_pairs):,} interactions / batch_size {batch_size})\n")
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Recall@20':>10} | {'NDCG@20':>9}")
    print("-" * 45)

    best_recall, best_ndcg = 0.0, 0.0

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0

        for _ in range(n_batches):
            users, pos_items, neg_items = _sample_batch(
                dataset.train_user_items, dataset.n_items, batch_size
            )
            users     = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            optimizer.zero_grad()
            loss = _bpr_loss(model, users, pos_items, neg_items, reg_lambda)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % eval_every == 0:
            avg_loss = total_loss / n_batches
            recall, ndcg = recall_and_ndcg_at_k(
                model, dataset.train_user_items, dataset.test_user_items, k=20
            )
            marker = " ←" if recall > best_recall else ""
            if recall > best_recall:
                best_recall, best_ndcg = recall, ndcg
            print(f"{epoch:>6} | {avg_loss:>8.4f} | {recall:>10.4f} | {ndcg:>9.4f}{marker}")

    print("-" * 45)
    print(f"\nBest  Recall@20 : {best_recall:.4f}   (paper: 0.1823)")
    print(f"Best  NDCG@20   : {best_ndcg:.4f}   (paper: 0.1554)")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}\n")

    # local path
    data_path = Path(__file__).parent.parent / "data" / "gowalla"

    # Colab path override
    colab_path = Path("/content/drive/MyDrive/ZED_project_data/gowalla")
    if colab_path.exists():
        data_path = colab_path

    train(
        data_path   = data_path,
        emb_dim     = 64,
        n_layers    = 3,
        lr          = 0.001,
        reg_lambda  = 1e-4,
        batch_size  = 1024,
        n_epochs    = 200,
        eval_every  = 5,
        device      = device,
    )
