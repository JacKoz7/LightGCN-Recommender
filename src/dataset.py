import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path


class GowallaDataset:
    """Loads the Gowalla dataset and builds the normalized adjacency matrix
    used by LightGCN (Eq. 6-7 in the paper)."""

    def __init__(self, data_path):
        data_path = Path(data_path)
        self.train_user_items, self.train_pairs = self._load(data_path / "train.txt")
        self.test_user_items, _                 = self._load(data_path / "test.txt")

        self.n_users = max(
            max(self.train_user_items), max(self.test_user_items)
        ) + 1
        self.n_items = max(
            item for items in self.train_user_items.values() for item in items
        ) + 1

        self.norm_adj = self._build_norm_adj()

    def _load(self, filepath):
        user_items = {}
        pairs = []
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                user_id  = int(parts[0])
                item_ids = [int(x) for x in parts[1:]]
                user_items[user_id] = item_ids
                for item_id in item_ids:
                    pairs.append((user_id, item_id))
        return user_items, pairs

    def _build_norm_adj(self):
        # Step 1: build sparse interaction matrix R of shape (n_users x n_items)
        users = [u for u, _ in self.train_pairs]
        items = [i for _, i in self.train_pairs]
        data  = np.ones(len(self.train_pairs))
        R = sp.csr_matrix((data, (users, items)), shape=(self.n_users, self.n_items))

        # Step 2: build full adjacency matrix A = [[0, R], [R^T, 0]]  (Eq. 6)
        upper = sp.hstack([sp.csr_matrix((self.n_users, self.n_users)), R])
        lower = sp.hstack([R.T, sp.csr_matrix((self.n_items, self.n_items))])
        A = sp.vstack([upper, lower]).tocsr()

        # Step 3: symmetric normalization  Ã = D^(-1/2) A D^(-1/2)  (Eq. 7)
        degree    = np.array(A.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj   = D_inv_sqrt @ A @ D_inv_sqrt

        return self._sparse_to_tensor(norm_adj.tocoo())

    def _sparse_to_tensor(self, coo):
        indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
        values  = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape), check_invariants=False)
