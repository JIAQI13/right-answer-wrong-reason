"""Run a HuggingFace causal-LM (DeepSeek-R1 distilled) on every prompt
variant of the benchmark.

WHY DeepSeek-R1-Distill:
  - Reasoning models wrap their internal thought process in <think>...</think>
  - This allows us to separate reasoning from the visible answer
  - Distilled versions are smaller and faster than the full DeepSeek-R1

Generation strategy:
  - temperature=0: deterministic generation for reproducibility
  - batch_size=1: variable-length <think> blocks make batching inefficient
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

    with out_path.open("w") as f:
        for pid, cond, ht, prompt in tqdm(variants, desc=f"generate · {mcfg['name']}"):
            # Format as chat conversation.
            # WHY chat template: Models are trained with specific conversation
            # formats. Using the wrong format can degrade performance.
            messages = [{"role": "user", "content": prompt}]

            # apply_chat_template may return a BatchEncoding or a tensor;
            # normalize to a plain tensor on the correct device.
            input_ids = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            if hasattr(input_ids, "input_ids"):
                input_ids = input_ids.input_ids
            input_ids = input_ids.to(model.device)

            # Generate with temperature=0 (deterministic).
            # WHY deterministic: Reproducibility is critical for research.
            # We want the same prompt to always produce the same response.
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids,
                    max_new_tokens=mcfg.get("max_new_tokens", 1024),
                    temperature=mcfg.get("temperature", 0.0),
                    do_sample=mcfg.get("temperature", 0.0) > 0,
                    top_p=mcfg.get("top_p", 1.0),
                    pad_token_id=tok.pad_token_id,
                )

            # Decode only the NEW tokens (not the input prompt)
            response = tok.decode(out_ids[0, input_ids.shape[1]:], skip_special_tokens=False)
            response = clean_byte_bpe(response)

            # Split into thinking (internal) and visible (answer)
            thinking, visible = split_thinking(response, open_tag, close_tag)

            # Write full record
            f.write(json.dumps({
                "problem_id": pid,
                "condition": cond,
                "hint_type": ht,
                "prompt": prompt,
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
