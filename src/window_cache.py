"""Bounded per-layer KV storage for the manual inference path."""
import torch


class WindowCache:
    def __init__(self, budget: int):
        if budget < 1:
            raise ValueError("Cache budget must be positive")
        # Keep separate storage for each layer.
        self.budget = budget
        self.layers = {}

    def update(self, keys, values, layer_idx):
        # Tensor layout is [batch, KV heads, tokens, head dimension].
        if layer_idx in self.layers:
            old_keys, old_values = self.layers[layer_idx]
            keys = torch.cat((old_keys, keys), dim=2)
            values = torch.cat((old_values, values), dim=2)
        if keys.shape[2] > self.budget:
            # Clone the suffix: a slice alone would retain the full backing storage.
            keys = keys[:, :, -self.budget:, :].clone() # -self.budget selects the newest entries.
            values = values[:, :, -self.budget:, :].clone()
        self.layers[layer_idx] = (keys, values)
        return keys, values

    def get_seq_length(self):
        return self.layers[0][0].shape[2] if self.layers else 0

    def storage_bytes(self):
        # Count both K and V in every layer, before temporary GQA expansion.
        return sum(t.numel() * t.element_size()
                   for pair in self.layers.values() for t in pair)
