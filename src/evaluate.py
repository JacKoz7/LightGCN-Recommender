import torch
import numpy as np


def recall_and_ndcg_at_k(model, train_user_items, test_user_items, k=20, batch_size=2048):
    """Compute Recall@K and NDCG@K using the all-ranking protocol from the paper.

    For each test user: score ALL items, mask out training items, take top-K.
    This matches the evaluation protocol described in Section 4.1 of the paper.
    """
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model()

    user_ids = list(test_user_items.keys())
    recalls, ndcgs = [], []

    for start in range(0, len(user_ids), batch_size):
        batch_users = user_ids[start:start + batch_size]
        scores = user_emb[batch_users] @ item_emb.T  # (B, n_items)

        for idx, user in enumerate(batch_users):
            # mask items seen during training so they cannot be recommended
            train_items = train_user_items.get(user, [])
            scores[idx][train_items] = -float("inf")

            top_k      = torch.topk(scores[idx], k).indices.cpu().numpy()
            true_items = set(test_user_items[user])

            hits = [1 if item in true_items else 0 for item in top_k]

            recall = sum(hits) / len(true_items)
            recalls.append(recall)

            dcg   = sum(h / np.log2(rank + 2) for rank, h in enumerate(hits))
            idcg  = sum(1 / np.log2(rank + 2) for rank in range(min(len(true_items), k)))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(recalls)), float(np.mean(ndcgs))
