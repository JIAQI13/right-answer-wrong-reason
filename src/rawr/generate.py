"""Run a HuggingFace causal-LM (DeepSeek-R1 distilled) on every prompt
variant of the benchmark.

WHY DeepSeek-R1-Distill:
  - Reasoning models wrap their internal thought process in <think>...</think>
  - This allows us to separate reasoning from the visible answer
  - Distilled versions are smaller and faster than the full DeepSeek-R1

Generation strategy:
  - temperature=0: deterministic generation for reproducibility
  - batch_size=8: process multiple prompts in parallel for GPU utilization
  - chat template: models are trained with specific conversation formats

Output: responses.jsonl with fields:
  - problem_id, condition, hint_type
  - prompt, response (full model output)
  - thinking_text (extracted from <think> tags)
  - visible_text (everything after </think>)
  - extracted_answer (parsed from visible_text)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable

import torch
import yaml
from tqdm import tqdm


# Regex for extracting the final answer from visible text.
# Matches patterns like "Answer: A", "answer = (B)", etc.
# Case-insensitive, matches the LAST occurrence (in case model discusses multiple).
ANSWER_RE = re.compile(
    r"answer\s*[:=]\s*\(?\s*([A-Da-d])\s*\)?\b",
    re.IGNORECASE,
)

# Fast tokenizer byte-fallback characters that sometimes leak through
# when decoder chain is incomplete. These are GPT-2 / Llama style:
#   Ġ = space (U+0120), Ċ = \n (U+010A), č = \r (U+010D)
# WHY clean these: They're artifacts of the tokenizer, not actual text.
_BYTE_FALLBACK = {
    "\u0120": " ",   # Ġ
    "\u010a": "\n",  # Ċ
    "\u010d": "\r",  # č
}


def clean_byte_bpe(text: str) -> str:
    """Force-clean byte-BPE fallback characters that tokenizer.decode()
    sometimes leaves behind (fast tokenizer + decoder config edge cases).

    Idempotent: safe to call on already-clean text.
    """
    for bad, good in _BYTE_FALLBACK.items():
        text = text.replace(bad, good)
    return text


def split_thinking(text: str, open_tag: str, close_tag: str) -> tuple[str, str]:
    """Split model output into (thinking_text, visible_text).

    For DeepSeek-R1 outputs the pattern is `<think>...</think><visible>`.

    WHY we need this:
      - thinking_text = the model's internal reasoning process
      - visible_text = the answer the model shows to the user
      - We analyze both: thinking for hint citation, visible for answer extraction

    Edge cases handled:
      - No tags at all: treat everything as visible
      - Only close tag (chat template auto-opens thinking): split on close tag
      - Nested tags: take the LAST <think> block (most recent reasoning)
    """
    if open_tag in text and close_tag in text:
        # take the LAST <think>..</think> block, in case the model nests
        i = text.rfind(open_tag)
        j = text.find(close_tag, i)
        if j > i:
            return text[i + len(open_tag): j].strip(), text[j + len(close_tag):].strip()
    # Some R1-distill checkpoints omit the opening <think> tag and emit
    # only the closing one when chat templates auto-open thinking mode.
    if close_tag in text and open_tag not in text:
        j = text.find(close_tag)
        return text[:j].strip(), text[j + len(close_tag):].strip()
    return "", text.strip()


def extract_answer(visible: str) -> str | None:
    """Extract the final answer letter (A/B/C/D) from the visible text.

    Priority:
      1. Explicit "Answer: X" pattern (most reliable)
      2. Last "(X)" pattern in the final 300 chars (fallback)

    WHY two strategies:
      - "Answer: X" is what we instructed the model to produce
      - "(X)" fallback handles cases where model doesn't follow instructions
      - Looking at the END of visible text: the model might discuss multiple
        answers during reasoning before committing to one
    """
    if not visible:
        return None
    # 1) Look for the explicit "Answer: X" line first
    matches = ANSWER_RE.findall(visible)
    if matches:
        return matches[-1].upper()
    # 2) Fallback: last "(X)" pattern in the final portion
    m = re.findall(r"\(([A-Da-d])\)", visible[-300:])
    return m[-1].upper() if m else None


def iter_variants(records: list[dict]) -> Iterable[tuple[str, str, str | None, str]]:
    """Yield (problem_id, condition, hint_type, prompt) for all variants.

    Flat iteration makes the generation loop simple and readable.
    """
    for r in records:
        if "prompt" in r["variants"].get("clean", {}):
            yield r["problem_id"], "clean", None, r["variants"]["clean"]["prompt"]
        for cond in ("faithful_hint", "misleading_hint"):
            for ht, slot in r["variants"].get(cond, {}).items():
                yield r["problem_id"], cond, ht, slot["prompt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total prompts (for dry runs)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override batch size from config")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    mcfg, gcfg = cfg["model"], cfg["generate"]
    if args.model:
        mcfg["name"] = args.model

    # Skip local generation if using precomputed responses
    if gcfg.get("use_young_precomputed", False):
        print("[info] use_young_precomputed=true — skipping local generation.")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[info] loading model {mcfg['name']}")
    tok = AutoTokenizer.from_pretrained(
        mcfg["name"], trust_remote_code=mcfg.get("trust_remote_code", False),
    )
    # Some models don't set pad_token; use eos_token as fallback.
    # WHY we need pad_token: generate() requires it for batch generation.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Set padding side to LEFT for decoder-only generation.
    # WHY left padding: In decoder-only models, the model predicts the next token
    # based on the left context. With right padding, padding tokens would be
    # in the "future" of the sequence, which can confuse the model.
    tok.padding_side = "left"

    # Map config dtype string to torch dtype
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(mcfg.get("dtype", "bfloat16"), torch.bfloat16)
    device = mcfg.get("device", "auto")

    # Cross-platform device auto-detection:
    #   CUDA (NVIDIA) → cuda
    #   MPS (Apple Silicon) → mps
    #   CPU fallback → cpu
    # WHY auto-detection: Makes the pipeline work on different hardware
    # without manual configuration.
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # MPS does not support bfloat16 before PyTorch 2.4; fall back to float16.
    # WHY: Apple Silicon's Metal API has limited dtype support.
    if device == "mps" and dtype == torch.bfloat16:
        print("[warn] MPS does not support bfloat16; falling back to float16")
        dtype = torch.float16

    print(f"[info] using device={device}, dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["name"],
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=mcfg.get("trust_remote_code", False),
    )
    model.eval()

    # Load benchmark
    bench_path = Path(cfg["benchmark"]["output_path"])
    if not bench_path.exists():
        raise SystemExit(f"missing benchmark file {bench_path}; run scripts/01_build_benchmark.py first")

    records = [json.loads(line) for line in bench_path.open()]
    out_dir = Path(gcfg["output_dir"]) / mcfg["name"].replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = list(iter_variants(records))
    if args.limit:
        variants = variants[: args.limit]
    out_path = out_dir / "responses.jsonl"

    open_tag = mcfg.get("thinking_open", "<think>")
    close_tag = mcfg.get("thinking_close", "</think>")

    # Batch size: config default is 8, but allow CLI override
    # WHY batch_size=8: Good balance between GPU utilization and memory usage.
    # For RTX 3080 (8GB) with 1.5B model, 8-16 prompts fit comfortably.
    # For 7B/8B models, reduce to 2-4.
    batch_size = args.batch_size if args.batch_size else gcfg.get("batch_size", 8)

    # Print progress info for Windows users
    print(f"[info] Starting generation: {len(variants)} prompts")
    print(f"[info] Using device: {device}")
    print(f"[info] Batch size: {batch_size}")
    print(f"[info] Progress bar will update after each batch completes")
    print("-" * 60)

    with out_path.open("w") as f:
        # Process in batches for parallel GPU utilization
        # WHY batching: GPUs are SIMD devices - they process multiple operations
        # in parallel. Running 1 prompt at a time wastes ~90% of GPU capacity.
        # With batch_size=8, we get ~8x throughput for ~same memory usage.
        for batch_start in tqdm(range(0, len(variants), batch_size),
                                desc=f"generate · {mcfg['name']}", ncols=80):
            batch = variants[batch_start:batch_start + batch_size]

            # Tokenize all prompts in batch with padding
            # WHY padding: All sequences in a batch must have the same length.
            # We pad to the longest sequence in the batch (not globally) to
            # minimize wasted computation.
            messages_list = [[{"role": "user", "content": prompt}]
                            for (_, _, _, prompt) in batch]

            # apply_chat_template for each prompt, then pad as a batch
            # We do this individually first because apply_chat_template may
            # add special tokens that vary by prompt (e.g., <think> auto-insertion)
            input_ids_list = []
            for messages in messages_list:
                encoded = tok.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt",
                )
                if hasattr(encoded, "input_ids"):
                    input_ids_list.append(encoded.input_ids[0])
                else:
                    input_ids_list.append(encoded[0])

            # Pad the batch to same length (left padding for decoder-only)
            # WHY left padding: For decoder-only generation, the model attends
            # to tokens on the left. Right padding would put padding tokens
            # in the "future" which can confuse causal attention.
            max_len = max(ids.shape[0] for ids in input_ids_list)
            padded_ids = []
            attention_masks = []
            for ids in input_ids_list:
                pad_len = max_len - ids.shape[0]
                # Left pad: prepend pad tokens
                padded = torch.cat([
                    torch.full((pad_len,), tok.pad_token_id, dtype=ids.dtype),
                    ids
                ])
                # Attention mask: 1 for real tokens, 0 for padding
                mask = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long),
                    torch.ones(ids.shape[0], dtype=torch.long)
                ])
                padded_ids.append(padded)
                attention_masks.append(mask)

            input_ids = torch.stack(padded_ids).to(model.device)
            attention_mask = torch.stack(attention_masks).to(model.device)

            # Generate with temperature=0 (deterministic).
            # WHY deterministic: Reproducibility is critical for research.
            # We want the same prompt to always produce the same response.
            temp = mcfg.get("temperature", 0.0)
            do_sample = temp > 0

            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": mcfg.get("max_new_tokens", 1024),
                "do_sample": do_sample,
                "pad_token_id": tok.pad_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = temp
                gen_kwargs["top_p"] = mcfg.get("top_p", 1.0)

            with torch.no_grad():
                out_ids = model.generate(**gen_kwargs)

            # Decode each response in the batch
            # We decode only the NEW tokens (not the input prompt)
            for i, (pid, cond, ht, _) in enumerate(batch):
                # Find where the input ends (skip padding on the left)
                # The input_ids[i] has left padding, so we find the first non-pad token
                input_len = (input_ids[i] != tok.pad_token_id).sum().item()
                response = tok.decode(out_ids[i, input_len:], skip_special_tokens=False)
                response = clean_byte_bpe(response)

                # Split into thinking (internal) and visible (answer)
                thinking, visible = split_thinking(response, open_tag, close_tag)

                # Write full record
                f.write(json.dumps({
                    "problem_id": pid,
                    "condition": cond,
                    "hint_type": ht,
                    "prompt": batch[i][3],
                    "response": response,
                    "thinking_text": thinking,
                    "visible_text": visible,
                    "extracted_answer": extract_answer(visible),
                    "model": mcfg["name"],
                    "ts": time.time(),
                }) + "\n")

    print(f"✔ wrote {len(variants)} responses → {out_path}")


if __name__ == "__main__":
    main()
