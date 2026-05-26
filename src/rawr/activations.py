"""Extract residual-stream activations using TransformerLens for every
prompt variant.

WHY this approach:
  - We focus on the token position immediately AFTER the </think> tag.
  - This is where the model "has decided but hasn't spoken yet."
  - The residual at this position captures the model's internal decision
    state before it commits to a verbal answer.
  - If this position encodes shortcut evidence from misleading hints,
    we have internal proof of RAWR even when the final answer is correct.

Output: one .pt file per problem under
    results/activations/<model>/<problem_id>.pt
with keys "<condition>__<hint_type>__L<L>" -> tensor[d_model].
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import yaml
from tqdm import tqdm


def select_layers(spec, n_layers: int) -> list[int]:
    """Convert a layer specification to actual layer indices.

    WHY we support multiple specs:
      - Explicit list: for targeted analysis (e.g., [10, 14, 18, 22, 26])
      - "all": for comprehensive layer-wise analysis
      - "middle_third": semantic content tends to form in middle layers
    """
    if isinstance(spec, list):
        return [int(i) for i in spec if 0 <= int(i) < n_layers]
    if spec == "all":
        return list(range(n_layers))
    if spec == "middle_third":
        # Middle third typically contains semantic/decision content
        # Early layers: low-level features, token patterns
        # Late layers: near output projection
        a, b = n_layers // 3, (2 * n_layers) // 3
        return list(range(a, b))
    raise ValueError(f"bad layers spec {spec!r}")


# ============================================================================
# TransformerLens base model mapping
# ============================================================================
# DeepSeek-R1-Distill models use standard architectures (Qwen2, Llama) but
# with different weights. TransformerLens needs to know the ARCHITECTURE
# (not the specific weights) to correctly set up hooks and layer structure.
#
# WHY dynamic detection:
#   - Hardcoding "if 'qwen' in name" is fragile
#   - Model names don't always reflect architecture (e.g., "DeepSeek-R1-Distill"
#     could be based on Qwen2 OR Llama)
#   - Reading from hf_model.config gives us the ground truth
#
# Format: (model_type, num_hidden_layers, hidden_size) -> tl_base_model
# ============================================================================
TL_BASE_MODELS = {
    # Qwen2 series
    ("qwen2", 28, 1536): "Qwen/Qwen2-1.5B",      # 1.5B parameters
    ("qwen2", 28, 3584): "Qwen/Qwen2-7B",        # 7B parameters
    ("qwen2", 48, 5120): "Qwen/Qwen2-14B",       # 14B parameters
    # Llama series
    ("llama", 32, 4096): "meta-llama/Llama-3.1-8B",  # 8B parameters
}

# Fallback mapping by architecture type (when exact match not found)
TL_FALLBACK = {
    "qwen2": "Qwen/Qwen2-1.5B",
    "llama": "meta-llama/Llama-3.1-8B",
}


def get_tl_base_model(hf_model) -> str:
    """Dynamically select TransformerLens base model from loaded HF model config.

    Priority:
      1. Exact match by (model_type, n_layers, hidden_size)
      2. Same architecture type match (ignore size)
      3. Fallback by model_type

    WHY exact match first: Different sizes of the same architecture can have
    different layer structures or attention patterns. Exact match ensures
    TransformerLens uses the correct configuration.
    """
    model_type = getattr(hf_model.config, "model_type", "")
    n_layers = getattr(hf_model.config, "num_hidden_layers", 0)
    hidden_size = getattr(hf_model.config, "hidden_size", 0)

    # Try exact match first (architecture + size)
    key = (model_type, n_layers, hidden_size)
    if key in TL_BASE_MODELS:
        return TL_BASE_MODELS[key]

    # Try same architecture type (any size)
    for (mt, nl, hs), tl_model in TL_BASE_MODELS.items():
        if mt == model_type:
            print(f"[warn] no exact match for {model_type}(layers={n_layers}, hidden={hidden_size}), "
                  f"using {tl_model} as reference")
            return tl_model

    # Last resort: fallback mapping
    if model_type in TL_FALLBACK:
        print(f"[warn] using fallback base model for {model_type}")
        return TL_FALLBACK[model_type]

    raise ValueError(
        f"Unsupported model architecture: {model_type}. "
        f"Supported types: {list(TL_FALLBACK.keys())}. "
        f"Please add the model to TL_BASE_MODELS mapping in activations.py."
    )


def find_answer_position(model, full_tokens: torch.Tensor,
                         close_tag: str = "</think>") -> int:
    """Return the token position right AFTER the </think> tag.

    WHY this position is critical:
      - Before </think>: model is still reasoning (internal thought process)
      - At </think>+1: model has finished reasoning and is about to speak
      - After </think>+1: model is verbalizing the answer (already committed)

    This position captures the "decision state" - what the model has decided
    but hasn't said yet. This is the most informative position for detecting
    whether the model used shortcut reasoning.

    If the tag is absent, return the last position as a fallback.
    """
    text = model.to_string(full_tokens[0])
    idx = text.rfind(close_tag)
    if idx < 0:
        return full_tokens.shape[1] - 1
    # Re-tokenize prefix up to and including </think> to find the position.
    # We can't just use character index because tokenization is not 1:1 with characters.
    prefix = text[: idx + len(close_tag)]
    prefix_tokens = model.to_tokens(prefix, prepend_bos=False)
    pos = min(prefix_tokens.shape[1], full_tokens.shape[1] - 1)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    mcfg, acfg = cfg["model"], cfg["activations"]
    model_name = args.model or mcfg["name"]

    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[info] loading {model_name} into HF, then wrapping with TransformerLens")
    hf_tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=mcfg.get("trust_remote_code", False))
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if mcfg.get("dtype") == "bfloat16" else torch.float16,
        device_map="auto",
        trust_remote_code=mcfg.get("trust_remote_code", False),
    )

    # Detect architecture dynamically from the loaded model's config.
    # WHY: Model names can be misleading (e.g., "DeepSeek-R1-Distill" doesn't
    # tell us if it's Qwen2 or Llama based). The config is the source of truth.
    model_type = getattr(hf_model.config, "model_type", "unknown")
    n_layers = getattr(hf_model.config, "num_hidden_layers", 0)
    hidden_size = getattr(hf_model.config, "hidden_size", 0)
    print(f"[info] detected architecture: {model_type}, layers={n_layers}, hidden_size={hidden_size}")

    tl_base_model = get_tl_base_model(hf_model)
    print(f"[info] using TransformerLens config from {tl_base_model}")

    # Wrap HF model with TransformerLens.
    # WHY fold_ln=False, center_writing_weights=False, center_unembed=False:
    #   - These defaults in TransformerLens normalize activations for interpretability.
    #   - We disable them to keep activations as close as possible to the original model.
    #   - This ensures our probes and distance metrics are computed on the
    #     actual activation distribution, not a normalized version.
    #
    # WHY local_files_only=True + try-except fallback:
    #   - HookedTransformer.from_pretrained() tries to download config from HF even
    #     when hf_model is provided. This fails in network-restricted environments
    #     (e.g., China using hf-mirror.com).
    #   - First try with local_files_only=True (use cache if available)
    #   - If that fails, use from_pretrained_no_processing() which uses the HF
    #     model's own config directly, no network access needed.
    dtype = torch.bfloat16 if mcfg.get("dtype") == "bfloat16" else torch.float16
    try:
        model = HookedTransformer.from_pretrained(
            tl_base_model,
            hf_model=hf_model,
            tokenizer=hf_tok,
            dtype=dtype,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            local_files_only=True,
        )
    except Exception as e:
        print(f"[warn] from_pretrained failed ({e}), trying from_pretrained_no_processing...")
        model = HookedTransformer.from_pretrained_no_processing(
            tl_base_model,
            hf_model=hf_model,
            tokenizer=hf_tok,
            dtype=dtype,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            local_files_only=True,
        )
    del hf_model  # Free memory - we don't need the HF model anymore
    model.eval()

    n_layers = model.cfg.n_layers
    layers = select_layers(acfg["layers"], n_layers)
    hook_keys = [f"blocks.{L}.{acfg['hook_name']}" for L in layers]
    print(f"[info] capturing {len(hook_keys)} hooks at layers {layers}")

    # Read the responses file to get each variant's actual generation
    # (including the model's <think>...</think>).
    # WHY re-feed the full prompt+thinking:
    #   - We need to extract activations from the SAME forward pass that
    #     produced the answer.
    #   - The model's internal reasoning affects the residual stream at
    #     the answer-token position.
    #   - If we just fed the prompt without the thinking, we'd be measuring
    #     a different (hypothetical) forward pass.
    gen_path = (Path(cfg["generate"]["output_dir"]) /
                model_name.replace("/", "__") / "responses.jsonl")
    if not gen_path.exists():
        raise SystemExit(f"missing {gen_path}; run scripts/02_generate.py first")

    by_variant: dict = {}
    for line in gen_path.open():
        r = json.loads(line)
        by_variant[(r["problem_id"], r["condition"], r["hint_type"])] = r

    bench_path = Path(cfg["benchmark"]["output_path"])
    records = [json.loads(l) for l in bench_path.open()]
    if args.limit:
        records = records[: args.limit]

    out_dir = Path(acfg["output_dir"]) / model_name.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    close_tag = mcfg.get("thinking_close", "</think>")

    for rec in tqdm(records, desc="extract activations"):
        # WHY per-problem bundles:
        #   - All variants of the same problem are stored together.
        #   - This makes it easy to compare clean vs misleading residuals
        #     for the SAME problem (paired analysis).
        #   - Also allows selective re-analysis without re-extracting everything.
        bundle: dict[str, torch.Tensor] = {}
        flat = []

        # Build flat list of (condition, hint_type, prompt) for iteration
        if "prompt" in rec["variants"].get("clean", {}):
            flat.append(("clean", "none", rec["variants"]["clean"]["prompt"]))
        for cond in ("faithful_hint", "misleading_hint"):
            for ht, slot in rec["variants"].get(cond, {}).items():
                flat.append((cond, ht, slot["prompt"]))

        for cond, ht, prompt in flat:
            ht_key = ht if ht is not None else "none"
            r = by_variant.get((rec["problem_id"], cond, ht_key if cond != "clean" else None))
            think = r.get("thinking_text", "") if r else ""
            visible = r.get("visible_text", "") if r else ""

            # Reconstruct the full text that the model actually saw+generated.
            # WHY: We need to run the EXACT same forward pass to get the
            # same residual stream at the answer-token position.
            full_text = (
                prompt + f"\n\n{close_tag.replace('</', '<')}\n{think}\n{close_tag}\n{visible}"
                if think else prompt
            )
            tokens = model.to_tokens(full_text)
            pos = find_answer_position(model, tokens, close_tag=close_tag)

            # Run with cache to capture activations at specified layers.
            # WHY torch.no_grad(): We're doing inference, not training.
            # Disabling gradient computation saves memory and compute.
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens, names_filter=hook_keys, return_type=None,
                )

            # Extract the residual at the answer-token position for each layer.
            # WHY cache[k][0, pos, :]:
            #   - Dimension 0: batch (we use batch_size=1)
            #   - Dimension 1: sequence position
            #   - Dimension 2: hidden state (d_model)
            # We want the single token at position `pos`.
            for L, k in zip(layers, hook_keys):
                act = cache[k][0, pos, :].detach().to(torch.float32).cpu()
                bundle[f"{cond}__{ht_key}__L{L}"] = act
            del cache  # Free memory before next iteration

        # Save per-problem bundle
        torch.save(bundle, out_dir / f"{rec['problem_id']}.pt")

    print(f"✔ activations saved under {out_dir}")


if __name__ == "__main__":
    main()
