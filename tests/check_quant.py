"""Test int8 reconstruction, append behavior, edge cases, and storage accounting."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from quant_cache import QuantCache

# Known random data includes independent heads and tokens with very different ranges.
torch.manual_seed(0)
for dtype in (torch.float16, torch.float32):
    x = torch.randn(1, 4, 5, 64).to(dtype)
    x[:, :, 0, :] = 0
    x[:, :, 1, :] *= 0.0001
    x[:, :, 2, :] *= 100
    q, scale = QuantCache.quantize(x)
    restored = QuantCache.dequantize(q, scale, dtype)
    assert q.dtype == torch.int8 and scale.dtype == torch.float32
    assert scale.shape == (1, 4, 5, 1)
    assert torch.isfinite(restored).all()
    assert torch.equal(restored[:, :, 0, :], x[:, :, 0, :])
    # Rounding to the nearest integer costs at most half a step, plus output-dtype rounding.
    allowance = scale / 2 + torch.finfo(dtype).eps * x.float().abs() + 1e-7
    assert ((restored.float() - x.float()).abs() <= allowance).all()
    for constant in (-3., 0., 3.):
        data = torch.full((1, 1, 2, 64), constant, dtype=dtype)
        a, b = QuantCache.quantize(data)
        assert torch.allclose(QuantCache.dequantize(a, b, dtype), data)

    cache = QuantCache()
    cache.store(x, -x, 0)
    old_k = cache.layers[0][0].clone()
    old_scale = cache.layers[0][2].clone()
    new = torch.ones(1, 4, 1, 64, dtype=dtype)
    k, v = cache.update(new, -new, 0)
    assert cache.get_seq_length() == 6
    assert k.shape == v.shape == (1, 4, 6, 64)
    assert k.dtype == v.dtype == dtype
    assert torch.allclose(k[:, :, -1, :], new[:, :, 0, :])
    assert torch.allclose(v[:, :, -1, :], -new[:, :, 0, :])
    # Appending must neither change old quantized values/scales nor retain full-precision K/V.
    assert torch.equal(cache.layers[0][0][:, :, :5, :], old_k)
    assert torch.equal(cache.layers[0][2][:, :, :5, :], old_scale)
    cache.store(new, new, 1)
    assert cache.layers[0][0].shape[2] == 6 and cache.layers[1][0].shape[2] == 1
    assert cache.storage_bytes() == 2 * 4 * 7 * 64
    assert cache.metadata_bytes() == 2 * 4 * 7 * 4
    for layer in cache.layers.values():
        assert [t.dtype for t in layer] == [torch.int8, torch.int8, torch.float32, torch.float32]
        for tensor in layer:
            assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
print('PASS: int8 error bounds, zeros, constants, fp16/fp32, append, layers, actual storage')
