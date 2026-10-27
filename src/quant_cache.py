"""Store K/V as symmetric int8 values plus per-token, per-head scales."""
import torch


class QuantCache:
    """Compress numeric precision without dropping tokens or changing RoPE positions.

    Each layer stores (int8 K, int8 V, float32 K scales, float32 V scales).
    Scales have shape [batch, KV heads, tokens, 1] and broadcast across head_dim.
    Full-precision reconstructions are temporary attention inputs, not stored here.
    """
    def __init__(self):
        # dict that holds each layer
        self.layers = {}

    @staticmethod
    def quantize(tensor):
        """Map each token/head vector to [-127, 127] using its largest magnitude. --> quantizes the values to int8"""
        # Compute scales in float32 so small fp16 values do not underflow during division.
        data = tensor.detach().float()
        scale = data.abs().amax(dim=-1, keepdim=True) / 127.0
        # An all-zero vector needs a nonzero divisor; its reconstructed values remain zero.
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        values = (data / scale).round().clamp(-127, 127).to(torch.int8)
        return values, scale

    @staticmethod
    def dequantize(values, scale, dtype):
        #"Undo" of the compression back to float32
        # Reconstruct in float32, then return the model's attention dtype (usually fp16).
        return (values.float() * scale).to(dtype)

    def store(self, keys, values, layer_idx):
        """Quantize only new entries; previous entries keep their original scales."""
        qk, sk = self.quantize(keys)
        qv, sv = self.quantize(values)
        if layer_idx in self.layers:
            old_k, old_v, old_sk, old_sv = self.layers[layer_idx]
            # Append along the token axis. Old K/V never undergo repeated requantization.
            qk = torch.cat((old_k, qk), dim=2)
            qv = torch.cat((old_v, qv), dim=2)
            sk = torch.cat((old_sk, sk), dim=2)
            sv = torch.cat((old_sv, sv), dim=2)
        self.layers[layer_idx] = (qk, qv, sk, sv)

    def update(self, keys, values, layer_idx):
        """Append compressed entries and temporarily reconstruct K/V for attention."""
        self.store(keys, values, layer_idx)
        qk, qv, sk, sv = self.layers[layer_idx]
        return (self.dequantize(qk, sk, keys.dtype),
                self.dequantize(qv, sv, values.dtype))

    def get_seq_length(self):
        return self.layers[0][0].shape[2] if self.layers else 0

    def storage_bytes(self):
        # Quantized K/V payload only; scales are counted separately by metadata_bytes().
        return sum(t.numel() * t.element_size()
                   for layer in self.layers.values() for t in layer[:2])

    def metadata_bytes(self):
        return sum(t.numel() * t.element_size()
                   for layer in self.layers.values() for t in layer[2:])
