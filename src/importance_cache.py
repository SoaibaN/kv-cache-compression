"""Keep tokens with the highest cumulative attention scores in each layer."""
import torch
from window_cache import WindowCache


class ImportanceCache(WindowCache):
    """Reuse window-cache bookkeeping, but select tokens by attention rather than age.

    Each layer has one score per cached token. Scores accumulate while a token is retained.

    Evicted tokens and their scores cannot be recovered.

    No score decay or reserved recent-token slots are used, so older tokens can have an advantage.
    """
    def __init__(self, budget: int):
        # The parent validates the budget and creates per-layer K/V storage.
        super().__init__(budget)
        # scores are separate metadata, not model keys or values.
        self.scores = {}

    def update(self, keys, values, layer_idx):
        """Append incoming [batch, KV heads, tokens, head_dim] tensors for attention."""
        # Delay eviction until attention has scored the incoming tokens.
        # This temporarily allows the full prompt, or budget+1 tokens during decode.
        new_scores = torch.zeros(keys.shape[2], dtype=torch.float32, device=keys.device)
        if layer_idx in self.layers:
            old_keys, old_values = self.layers[layer_idx]
            keys = torch.cat((old_keys, keys), dim=2)
            values = torch.cat((old_values, values), dim=2)
            # Existing scores keep their history, new tokens start at zero.
            new_scores = torch.cat((self.scores[layer_idx], new_scores))
        self.layers[layer_idx] = (keys, values)
        self.scores[layer_idx] = new_scores
        return keys, values

    def record_attention(self, weights, layer_idx):
        """Accumulate attention and prune after the current attention output is computed."""

        # Average over batch/query heads and accumulate attention from all query positions.
        # Current inference uses batch size one, all heads share the selected token indices.
        received = weights.detach().float().mean(dim=(0, 1)).sum(dim=0)
        scores = self.scores[layer_idx] + received
        keys, values = self.layers[layer_idx]
        if scores.numel() > self.budget:
            # Stable ties favor earlier entries.
            keep = torch.argsort(scores, descending=True, stable=True)[:self.budget]
            keep = keep.sort().values # sort the indices that were kept to be in order
            # Use identical indices for K, V, and scores. index_select allocates compact storage.
            # do not re-encode ROPE positions because Keys already contain them
            keys = keys.index_select(2, keep)
            values = values.index_select(2, keep)
            scores = scores.index_select(0, keep)
        self.layers[layer_idx] = (keys, values)
        self.scores[layer_idx] = scores

    def metadata_bytes(self):
        """Return score-tensor bytes, in addition to inherited storage_bytes() for K/V."""
        # Report score storage separately from the required K/V byte total.
        return sum(t.numel() * t.element_size() for t in self.scores.values())
