import torch
import torch.nn as nn


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, n_layers, norm_adj):
        super().__init__()
        self.n_users  = n_users
        self.n_items  = n_items
        self.n_layers = n_layers
        self.norm_adj = norm_adj  # Ã — sparse tensor (n_users+n_items) × (n_users+n_items)

        # The only trainable parameters in the entire model (Eq. 4 — 0th layer embeddings)
        self.embedding = nn.Embedding(n_users + n_items, emb_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self):
        # E^(0): one embedding vector per user and per item
        E = self.embedding.weight

        # Graph convolution: K times E^(k+1) = Ã · E^(k)  (Eq. 3)
        layer_embeddings = [E]
        for _ in range(self.n_layers):
            E = torch.sparse.mm(self.norm_adj, E)
            layer_embeddings.append(E)

        # Final embedding = uniform average over all layers  (Eq. 4, alpha_k = 1/(K+1))
        final_emb = torch.stack(layer_embeddings, dim=0).mean(dim=0)

        user_emb = final_emb[:self.n_users]
        item_emb = final_emb[self.n_users:]
        return user_emb, item_emb

    def predict(self, users, items):
        user_emb, item_emb = self.forward()
        return (user_emb[users] * item_emb[items]).sum(dim=-1)
