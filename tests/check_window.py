"""Check eviction contents, per-layer isolation, and actual backing storage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from window_cache import WindowCache

cache = WindowCache(3)
k = torch.arange(10, dtype=torch.float32).reshape(1, 1, 5, 2)
v = k + 100
cache.update(k, v, 0)
assert torch.equal(cache.layers[0][0], k[:, :, -3:, :])
assert torch.equal(cache.layers[0][1], v[:, :, -3:, :])
# Eviction must release storage, not just make the visible tensor smaller.
for t in cache.layers[0]:
    assert t.untyped_storage().nbytes() == t.numel() * t.element_size()
new = torch.tensor([[[[10., 11.]]]])
keys, values = cache.update(new, new + 100, 0)
assert torch.equal(keys, torch.tensor([[[[6., 7.], [8., 9.], [10., 11.]]]]))
assert torch.equal(values, keys + 100)
cache.update(k[:, :, :2, :], v[:, :, :2, :], 1)
assert cache.get_seq_length() == 3
assert cache.storage_bytes() == (3 + 2) * 2 * 2 * 4
# A one-token budget still keeps the current token.
small = WindowCache(1)
small.update(k, v, 0)
assert torch.equal(small.layers[0][0], k[:, :, -1:, :])
print('PASS: eviction, K/V alignment, storage size, layer isolation, budget 1')
