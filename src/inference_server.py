#!/usr/bin/env python3
"""TinyLlama inference with selectable KV-cache compression.

``--verbose`` unwraps the model into RMSNorm, Q/K/V projection, RoPE, GQA,
attention, and SwiGLU operations so cache behavior can be inspected directly.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import torch
from window_cache import WindowCache
from importance_cache import ImportanceCache
from quant_cache import QuantCache
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def kv_seq_len(kv_cache) -> int:
    if hasattr(kv_cache, "get_seq_length"):
        return kv_cache.get_seq_length()
    return kv_cache[0][0].shape[2]


def kv_bytes(n_layers, n_kv_heads, seq_len, head_dim, batch, itemsize) -> int:
    return 2 * itemsize * n_layers * n_kv_heads * seq_len * head_dim * batch


def stored_cache_bytes(compression, kv_cache, dense_bytes):
    """Return stored K/V bytes and auxiliary compression-metadata bytes.

    The uncompressed baseline uses the dense GQA formula. Compressed modes
    instead count the tensors that their cache objects actually retain.
    """
    if compression == "none":
        return dense_bytes, 0
    kv_data = kv_cache.storage_bytes()
    auxiliary = kv_cache.metadata_bytes() if hasattr(kv_cache, "metadata_bytes") else 0
    return kv_data, auxiliary


def decode_answer(tokenizer, token_ids) -> str:
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    for tag in ("<|system|>", "<|user|>", "<|assistant|>"):
        text = text.replace(tag, "")
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal full-server LLM inference")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Total new tokens, including the one chosen during prefill "
        "(the decode loop runs at most this many minus 1)",
    )
    parser.add_argument("--verbose", action="store_true", help="Unwrap model() layer by layer")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Use the explicit layer loop without enabling extra verbose checks",
    )
    parser.add_argument(
        "--quiet-tokens",
        action="store_true",
        help="Suppress token-by-token output during timed benchmark runs",
    )
    # Default to the unchanged baseline when no compression method is specified.
    parser.add_argument(
        "--compression",
        choices=("none", "window", "importance", "quant"),
        default="none",
        help="KV cache compression method (default: none; quant stores int8 K/V)",
    )
    parser.add_argument("--cache-budget", type=int, default=128,
                        help="Maximum retained tokens per layer for window/importance compression (default: 128)")
    args = parser.parse_args()

    if args.cache_budget < 1:
        parser.error("--cache-budget must be positive")
    # Compression uses the explicit layer loop even without verbose diagnostics.
    manual = args.manual or args.verbose or args.compression != "none"

    prompt = args.prompt
    if prompt is None:
        prompt = input("Prompt: ").strip()
    if not prompt:
        print("Empty prompt.", file=sys.stderr)
        sys.exit(1)

    dtype = torch.float16 if args.device != "cpu" else torch.float32

    # --- 1) Load model ---
    print("\n[1] Load model")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(args.device)
    model.eval()

    cfg = model.config
    d_model = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim = getattr(cfg, "head_dim", d_model // num_heads)
    attn_scale = 1.0 / math.sqrt(head_dim)
    kv_itemsize = 2 if dtype == torch.float16 else 4

    print(f"    model={args.model}")
    print(f"    device={args.device}, dtype={dtype}")
    print(f"    d_model={d_model}, layers={n_layers}, heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}")
    print(f"    GQA: {num_heads} query heads share {num_kv_heads} KV heads ({num_heads // num_kv_heads}x smaller KV cache than MHA)")
    print(f"    load time={(time.perf_counter() - t0) * 1000:.0f} ms")

    # --- 2) Receive prompt ---
    print("\n[2] Receive prompt")
    print(f"    raw: {prompt!r}")
    messages = [{"role": "user", "content": prompt}]
    chat_kwargs = {"add_generation_prompt": True}
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, **chat_kwargs)
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", **chat_kwargs
    ).to(args.device)
    print(f"    templated: {formatted!r}")
    print(f"    {input_ids.shape[1]} tokens")

    # --- 3) Prefill + KV cache ---
    print("\n[3] Prefill + KV cache")
    # CUDA kernels are asynchronous; synchronize at both timing boundaries.
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t_prefill = time.perf_counter()

    if not manual:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        kv_cache = outputs.past_key_values
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    else:
        print("    --- unwrap model(): layer-by-layer prefill ---")
        with torch.no_grad():
            # 1. Embed token IDs
            hidden_states = model.get_input_embeddings()(input_ids)
            batch_size, seq_len, _ = hidden_states.shape
            print(f"    * Token IDs {tuple(input_ids.shape)} -> embeddings {tuple(hidden_states.shape)}")

            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=args.device)).bool()
            position_ids = torch.arange(seq_len, device=args.device).unsqueeze(0)
            cache_types = {"window": WindowCache, "importance": ImportanceCache}
            if args.compression in cache_types:
                kv_cache = cache_types[args.compression](args.cache_budget)
            elif args.compression == "quant":
                # Numeric compression retains all tokens, so it has no eviction budget.
                kv_cache = QuantCache()
            else:
                kv_cache = DynamicCache()

            # 2. LOOP THROUGH ALL LAYERS MANUALLY
            for layer_idx, layer in enumerate(model.model.layers):
                attn, mlp = layer.self_attn, layer.mlp

                # A. Pre-Attention Normalization
                x_norm = layer.input_layernorm(hidden_states)

                # B. Attention Computations
                # Q, K, V Projections
                q = attn.q_proj(x_norm).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                k = attn.k_proj(x_norm).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
                v = attn.v_proj(x_norm).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

                # Apply Rotary Position Embeddings (RoPE)
                cos, sin = model.model.rotary_emb(v, position_ids)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                # Save K/V while keeping full causal attention for prefill.
                # Window trims now; importance selects tokens after attention scores exist.
                if args.compression == "quant":
                    # Prefill attention uses original K/V; store int8 copies for future decode.
                    # Avoid building an unused full-cache reconstruction during prefill.
                    kv_cache.store(k, v, layer_idx)
                else:
                    kv_cache.update(k, v, layer_idx)

                # Handle Grouped-Query Attention (GQA) expansion
                if num_kv_heads < num_heads:
                    k_exp = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
                    v_exp = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
                else:
                    k_exp, v_exp = k, v

                # Scaled Dot-Product Attention
                scores = torch.matmul(q, k_exp.transpose(-2, -1)) * attn_scale
                scores = scores.masked_fill(~causal_mask, float("-inf"))
                attn_weights = torch.softmax(scores, dim=-1)
                attn_output = torch.matmul(attn_weights, v_exp)
                if args.compression == "importance":
                    # Prefill scores use full causal attention before selecting retained tokens.
                    kv_cache.record_attention(attn_weights, layer_idx)

                # Output Projection + Residual Add
                attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
                hidden_states = hidden_states + attn.o_proj(attn_output)

                # C. SwiGLU MLP + Residual Add
                x_mid = layer.post_attention_layernorm(hidden_states)
                hidden_states = hidden_states + mlp.down_proj(mlp.act_fn(mlp.gate_proj(x_mid)) * mlp.up_proj(x_mid))

                if layer_idx == 0:
                    print(f"    * Layer 0 Q {tuple(q.shape)}  K cached {tuple(k.shape)}")
                    print(f"    * Layer 0 scores {tuple(scores.shape)} (pre-softmax)  weights {tuple(attn_weights.shape)}")

            print(f"    * Ran all {n_layers} layers manually")

            # 3. Final RMSNorm + LM head
            logits = model.lm_head(model.model.norm(hidden_states))
            next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t_prefill) * 1000

    if args.verbose:  # sanity check, outside the prefill timer
        with torch.no_grad():
            ref_id = model(input_ids=input_ids, use_cache=False, return_dict=True).logits[:, -1, :].argmax(dim=-1)
        print(f"    * First token matches HuggingFace model(): {ref_id.item() == next_token_id.item()}")
    if args.compression in ("window", "importance"):
        print(f"    retained KV bytes={kv_cache.storage_bytes()} budget={args.cache_budget}")
    seq = kv_seq_len(kv_cache)
    gqa = kv_bytes(n_layers, num_kv_heads, seq, head_dim, 1, kv_itemsize)
    mha = kv_bytes(n_layers, num_heads, seq, head_dim, 1, kv_itemsize)
    stored_kv, auxiliary = stored_cache_bytes(args.compression, kv_cache, gqa)
    first_text = tokenizer.decode(next_token_id[0], skip_special_tokens=False)
    print(f"    prefill time={prefill_ms:.1f} ms")
    print(f"    KV cache seq_len={seq}")
    if args.compression == "quant":
        payload, scales = kv_cache.storage_bytes(), kv_cache.metadata_bytes()
        print(f"    quantized KV bytes={payload} scale bytes={scales} total cache bytes={payload + scales}")
    else:
        print(f"    KV cache {gqa / 1024:.1f} KB (GQA) vs {mha / 1024:.1f} KB if MHA")
    print(
        f"    prefill stored KV bytes={stored_kv} auxiliary bytes={auxiliary} "
        f"total cache bytes={stored_kv + auxiliary}"
    )
    print(f"    first output token id={next_token_id.item()} piece={first_text!r}  (one word may take several tokens)")

    # --- 4) Autoregressive decode ---
    print("\n[4] Autoregressive decode")
    generated = [next_token_id.item()]
    # Absolute position of the next token. Track it separately: after eviction,
    # the cache length no longer equals the token's true position (RoPE).
    cur_pos = input_ids.shape[1]
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t_decode = time.perf_counter()
    eos_id = tokenizer.eos_token_id
    if not args.quiet_tokens:
        print(
            f"    prefill chose id={generated[0]:6d}  "
            f"text so far={tokenizer.decode(generated, skip_special_tokens=True)!r}"
        )

    # --max-tokens includes the prefill token, so decode emits at most one fewer.
    for step in range(args.max_tokens - 1):
        with torch.no_grad():
            if not manual:
                outputs = model(input_ids=next_token_id, past_key_values=kv_cache, use_cache=True, return_dict=True)
                kv_cache = outputs.past_key_values
                next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            else:
                # Same block as prefill, but Q has seq=1 and attends to the full cache
                position_ids = torch.tensor([[cur_pos]], device=args.device)
                hidden_states = model.get_input_embeddings()(next_token_id)
                batch_size = hidden_states.shape[0]

                # LOOP THROUGH ALL LAYERS MANUALLY
                for layer_idx, layer in enumerate(model.model.layers):
                    attn, mlp = layer.self_attn, layer.mlp

                    # A. Pre-Attention Normalization
                    x_norm = layer.input_layernorm(hidden_states)

                    # B. Attention Computations
                    # Q, K, V Projections
                    q = attn.q_proj(x_norm).view(batch_size, 1, num_heads, head_dim).transpose(1, 2)
                    k_new = attn.k_proj(x_norm).view(batch_size, 1, num_kv_heads, head_dim).transpose(1, 2)
                    v_new = attn.v_proj(x_norm).view(batch_size, 1, num_kv_heads, head_dim).transpose(1, 2)

                    # Apply Rotary Position Embeddings (RoPE)
                    cos, sin = model.model.rotary_emb(v_new, position_ids)
                    q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)

                    # Append K and V into the DynamicCache object
                    k_full, v_full = kv_cache.update(k_new, v_new, layer_idx)

                    # Handle Grouped-Query Attention (GQA) expansion
                    if num_kv_heads < num_heads:
                        k_exp = k_full.repeat_interleave(num_heads // num_kv_heads, dim=1)
                        v_exp = v_full.repeat_interleave(num_heads // num_kv_heads, dim=1)
                    else:
                        k_exp, v_exp = k_full, v_full

                    # Scaled Dot-Product Attention (no mask: the new token may attend to all cached positions)
                    scores = torch.matmul(q, k_exp.transpose(-2, -1)) * attn_scale
                    attn_weights = torch.softmax(scores, dim=-1)
                    attn_output = torch.matmul(attn_weights, v_exp)
                    if args.compression == "importance":
                        # Score the current attention, then prune storage for the next step.
                        kv_cache.record_attention(attn_weights, layer_idx)

                    # Output Projection + Residual Add
                    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, d_model)
                    hidden_states = hidden_states + attn.o_proj(attn_output)

                    # C. SwiGLU MLP + Residual Add
                    x_mid = layer.post_attention_layernorm(hidden_states)
                    hidden_states = hidden_states + mlp.down_proj(mlp.act_fn(mlp.gate_proj(x_mid)) * mlp.up_proj(x_mid))

                    if not args.quiet_tokens and step == 0 and layer_idx == 0:
                        print(f"    * decode step 1 layer 0: Q seq={q.shape[2]}  K seq={k_full.shape[2]}")

                # Final RMSNorm + LM head
                logits = model.lm_head(model.model.norm(hidden_states))
                next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        cur_pos += 1
        token_id = next_token_id.item()
        generated.append(token_id)
        if not args.quiet_tokens:
            piece = tokenizer.decode([token_id], skip_special_tokens=False)
            visible_text = tokenizer.decode(generated, skip_special_tokens=True)
            print(
                f"    step {step + 1:2d}: id={token_id:6d}  "
                f"piece={piece!r}  text so far={visible_text!r}  kv_len={kv_seq_len(kv_cache)}"
            )
        if token_id == eos_id:
            print("    (EOS — stopping decode)")
            break
        raw_text = tokenizer.decode(generated, skip_special_tokens=False)
        if "<|user|>" in raw_text or "<|system|>" in raw_text:
            print("    (chat role tag — stopping decode)")
            break

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - t_decode) * 1000

    # --- 5) Exit ---
    print("\n[5] Exit")
    print(f"    decode time={decode_ms:.1f} ms ({len(generated) - 1} new tokens)")
    print(f"    output token ids={generated}")
    print(f"    output: {decode_answer(tokenizer, generated)!r}")
    if args.compression in ("window", "importance"):
        print(f"    final retained KV bytes={kv_cache.storage_bytes()} kv_len={kv_seq_len(kv_cache)} next_position={cur_pos}")
    if args.compression == "importance":
        print(f"    score metadata bytes={kv_cache.metadata_bytes()}")
    if args.compression == "quant":
        payload, scales = kv_cache.storage_bytes(), kv_cache.metadata_bytes()
        print(f"    final quantized KV bytes={payload} scale bytes={scales} total cache bytes={payload + scales} "
              f"kv_len={kv_seq_len(kv_cache)} next_position={cur_pos}")
    final_seq = kv_seq_len(kv_cache)
    final_dense = kv_bytes(n_layers, num_kv_heads, final_seq, head_dim, 1, kv_itemsize)
    stored_kv, auxiliary = stored_cache_bytes(args.compression, kv_cache, final_dense)
    print(
        f"    final stored KV bytes={stored_kv} auxiliary bytes={auxiliary} "
        f"total cache bytes={stored_kv + auxiliary} kv_len={final_seq} next_position={cur_pos}"
    )
    print("    done.")


if __name__ == "__main__":
    main()
