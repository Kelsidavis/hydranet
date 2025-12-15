#!/usr/bin/env python3
"""Debug HydraNet forward pass step by step."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    import torch
    import torch.nn.functional as F
    from hydranet.v2.config import MixtralConfig, ExpertCacheConfig, KVCacheConfig
    from hydranet.v2.model.mixtral import OffloadedMixtral
    from hydranet.v2.model.loader import MixtralWeightLoader
    from hydranet.v2.cache.packed_expert_store import PackedExpertStore
    from transformers import AutoTokenizer

    device = torch.device("cuda")
    dtype = torch.float16
    model_path = "/home/k/models/mixtral-8x7b-instruct"
    packed_dir = Path("/home/k/models/mixtral-8x7b-instruct/packed_int4")

    print("=" * 60)
    print("DEBUG FORWARD PASS")
    print("=" * 60)

    # Config
    config = MixtralConfig()
    cache_config = ExpertCacheConfig(
        pinned_slots=0,
        hot_slots=1,
        probation_slots=1,
        enable_prefetch=False,
    )
    kv_config = KVCacheConfig()

    # Model
    print("\nLoading model...")
    model = OffloadedMixtral(
        config=config,
        cache_config=cache_config,
        kv_config=kv_config,
        device=device,
        dtype=dtype,
    )

    # Weights
    loader = MixtralWeightLoader(model_path, config, device, dtype)
    weights = loader.load_non_expert_weights()
    model.load_weights(weights)
    del weights

    # Packed store
    packed_store = PackedExpertStore(
        index_path=packed_dir / "experts.idx",
        bin_path=packed_dir / "experts.bin",
        device=device,
        use_mmap=True,
    )
    model.expert_cache.set_packed_store(packed_store)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model loaded!")

    # Single token test
    prompt = "[INST] Hi [/INST]"
    print(f"\nPrompt: {prompt}")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"Input IDs: {input_ids}")

    model.eval()
    with torch.no_grad():
        # Embedding
        hidden = model.embed_tokens(input_ids)
        print(f"\nAfter embedding:")
        print(f"  shape: {hidden.shape}")
        print(f"  mean: {hidden.mean().item():.6f}")
        print(f"  std: {hidden.std().item():.6f}")

        # Position IDs
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

        # Attention mask
        attention_mask = model._make_causal_mask(input_ids.shape[1], input_ids.shape[1])

        # First layer only
        layer = model.layers[0]

        # Attention
        residual = hidden
        hidden = layer.input_layernorm(hidden)
        print(f"\nAfter input norm (layer 0):")
        print(f"  mean: {hidden.mean().item():.6f}")
        print(f"  std: {hidden.std().item():.6f}")

        hidden_attn, _ = layer.self_attn(hidden, position_ids, attention_mask)
        hidden = residual + hidden_attn
        print(f"\nAfter attention (layer 0):")
        print(f"  mean: {hidden.mean().item():.6f}")
        print(f"  std: {hidden.std().item():.6f}")

        # MoE input
        residual = hidden
        moe_input = layer.post_attention_layernorm(hidden)
        print(f"\nMoE input (layer 0):")
        print(f"  mean: {moe_input.mean().item():.6f}")
        print(f"  std: {moe_input.std().item():.6f}")

        # Router
        router_out = layer.moe.router(moe_input, top_k_override=2)
        print(f"\nRouter output (layer 0):")
        print(f"  expert_indices: {router_out.expert_indices}")
        print(f"  expert_weights: {router_out.expert_weights}")
        print(f"  router_logits: {router_out.router_logits}")

        # Get first expert weights
        expert_idx = router_out.expert_indices[0, 0, 0].item()
        print(f"\nLoading expert {expert_idx} for layer 0...")
        expert_weights = model.expert_cache.get_expert_weights(0, expert_idx)

        print(f"  gate_proj shape: {expert_weights['gate_proj'].shape}")
        print(f"  gate_proj[0,0:5]: {expert_weights['gate_proj'][0,0:5]}")
        print(f"  gate_proj mean: {expert_weights['gate_proj'].mean().item():.6f}")
        print(f"  gate_proj std: {expert_weights['gate_proj'].std().item():.6f}")

        # Manual expert forward
        test_input = moe_input[0, 0:1, :]  # (1, hidden_dim)
        print(f"\nManual expert forward (single token):")
        print(f"  input shape: {test_input.shape}")
        print(f"  input mean: {test_input.mean().item():.6f}")

        gate = F.silu(F.linear(test_input, expert_weights["gate_proj"]))
        print(f"  gate output mean: {gate.mean().item():.6f}")

        up = F.linear(test_input, expert_weights["up_proj"])
        print(f"  up output mean: {up.mean().item():.6f}")

        gated = gate * up
        print(f"  gated mean: {gated.mean().item():.6f}")

        out = F.linear(gated, expert_weights["down_proj"])
        print(f"  expert output mean: {out.mean().item():.6f}")
        print(f"  expert output std: {out.std().item():.6f}")

        # Full MoE
        moe_out = layer.moe(moe_input, top_k=2)
        print(f"\nFull MoE output (layer 0):")
        print(f"  mean: {moe_out.mean().item():.6f}")
        print(f"  std: {moe_out.std().item():.6f}")

        hidden = residual + moe_out
        print(f"\nAfter layer 0:")
        print(f"  mean: {hidden.mean().item():.6f}")
        print(f"  std: {hidden.std().item():.6f}")

        # Full forward
        print("\n" + "=" * 40)
        print("Running full forward pass...")

        logits, _ = model.forward(input_ids, use_cache=False)
        print(f"\nFinal logits:")
        print(f"  shape: {logits.shape}")
        print(f"  mean: {logits.mean().item():.6f}")
        print(f"  std: {logits.std().item():.6f}")

        # Get top predictions for last token
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top_probs, top_ids = torch.topk(probs, k=10)
        print(f"\nTop 10 predictions for next token:")
        for i, (p, idx) in enumerate(zip(top_probs, top_ids)):
            token = tokenizer.decode([idx.item()])
            print(f"  {i+1}. '{token}' ({p.item():.4f})")

    # Cleanup
    model.expert_cache.shutdown()
    packed_store.close()


if __name__ == "__main__":
    main()
