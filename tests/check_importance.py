"""Use known attention weights to check score-based eviction and alignment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from importance_cache import ImportanceCache

cache = ImportanceCache(2)
# Token identity is encoded directly in K; V is K+10 so misalignment is easy to detect.
k = torch.tensor([[[[0.], [1.], [2.]]]])
cache.update(k, k + 10, 0)
cache.record_attention(torch.tensor([[[[.6, .1, .3]]]]), 0)
# Keep the important old token 0 rather than simply keeping recent tokens 1 and 2.
assert cache.layers[0][0].flatten().tolist() == [0., 2.]
assert cache.layers[0][1].flatten().tolist() == [10., 12.]
new = torch.tensor([[[[3.]]]])
# Updated scores become [0.7, 0.4, 0.8]; token 2 must lose to the new token 3.
cache.update(new, new + 10, 0)
cache.record_attention(torch.tensor([[[[.1, .1, .8]]]]), 0)
assert cache.layers[0][0].flatten().tolist() == [0., 3.]
assert torch.allclose(cache.scores[0], torch.tensor([.7, .8]))
# Two tokens x one float32 value x K/V = 16 bytes; two float32 scores = 8 bytes.
assert cache.storage_bytes() == 16 and cache.metadata_bytes() == 8
for t in (*cache.layers[0], cache.scores[0]):
    # A small tensor view backed by a larger allocation would fail this check.
    assert t.untyped_storage().nbytes() == t.numel() * t.element_size()
# An independent layer with tied scores retains earlier entries deterministically.
cache.update(k, k + 10, 1)
cache.record_attention(torch.ones(1, 2, 3, 3) / 3, 1)
assert cache.layers[1][0].flatten().tolist() == [0., 1.]
assert cache.layers[0][0].flatten().tolist() == [0., 3.]
small = ImportanceCache(1)
# The smallest valid budget must still select the highest-scoring entry.
small.update(k, k + 10, 0)
small.record_attention(torch.tensor([[[[.1, .8, .1]]]]), 0)
assert small.layers[0][0].item() == 1.
print('PASS: importance ranking, cumulative scores, aligned eviction, storage, ties, layer isolation, budget 1')
