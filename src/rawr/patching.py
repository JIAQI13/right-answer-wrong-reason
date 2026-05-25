"""Activation patching utility (H2 - causal evidence).

WHAT is activation patching:
  - Run the model on clean prompt, cache the residual at layer L
  - Run the model on misleading prompt, but at layer L, replace the residual
    with the clean version
  - If the answer flips from wrong→correct, layer L is causally responsible

WHY this gives causal evidence:
  - Correlation (H1 probe): "shortcut is represented at layer L"
  - Causation (H2 patching): "changing layer L changes the behavior"
  - Patching is the gold standard for "which layer caused what"

Strategy:
  - Replace the residual at the answer-token position (after </think>)
  - This is where the decision is encoded but not yet verbalized
  - If patching here flips the answer, this layer encoded the shortcut

Output: The patched generation (visible text after </think>).
Compare with original to see if answer changed.
"""
from __future__ import annotations

import torch


def find_pos(model, prompt: str, close_tag: str = "</think>") -> int:
    """Find the token position right AFTER the </think> tag.

    WHY this position:
      - Before </think>: model is still reasoning
      - At </think>+1: model has decided but hasn't spoken
      - After </think>+1: model is verbalizing (already committed)

    This is the optimal position for patching - we intercept the decision
    before it becomes the visible answer.
    """
    tokens = model.to_tokens(prompt)
    text = model.to_string(tokens[0])
    idx = text.rfind(close_tag)
    if idx < 0:
        return tokens.shape[1] - 1
    # Re-tokenize prefix up to and including </think>
    # Can't use character index directly because tokenization != character mapping
    prefix = text[: idx + len(close_tag)]
    return min(model.to_tokens(prefix, prepend_bos=False).shape[1], tokens.shape[1] - 1)


def patch_clean_into_misleading(
    model,
    clean_prompt: str,
    misleading_prompt: str,
    layer: int,
    hook_name: str = "hook_resid_post",
    max_new_tokens: int = 256,
    close_tag: str = "</think>",
) -> str:
    """Patch the clean residual into the misleading forward pass.

    Steps:
      1. Run clean prompt, cache residual at (layer, answer_token_pos)
      2. Run misleading prompt with a hook that replaces the residual
         at (layer, answer_token_pos) with the clean version
      3. Return the generated visible text

    WHY hook_resid_post:
      - This is the residual stream AFTER the transformer block
      - It's the "output" of the layer - what gets passed to the next layer
      - Patching here intercepts the layer's contribution to the final decision

    WHY clone the clean vector:
      - Prevents in-place modification issues
      - Ensures we have a clean copy for each patching operation
      - Device/dtype casting happens at patch time, not cache time
    """
    full_hook = f"blocks.{layer}.{hook_name}"

    # Step 1: Cache the clean residual
    clean_tokens = model.to_tokens(clean_prompt)
    target_pos = find_pos(model, clean_prompt, close_tag)

    # Run clean forward pass with cache
    # WHY return_type=None: We only want the cache, not the logits
    with torch.no_grad():
        _, cache = model.run_with_cache(clean_tokens, names_filter=[full_hook], return_type=None)

    # Extract the clean residual at the answer-token position
    # Dimensions: [batch, seq_pos, d_model] -> we want [0, target_pos, :]
    clean_vec = cache[full_hook][0, target_pos, :].clone()

    # Step 2: Prepare misleading prompt
    mis_tokens = model.to_tokens(misleading_prompt)
    mis_pos = find_pos(model, misleading_prompt, close_tag)

    # Define the patching hook
    # This function will be called during the forward pass at the specified layer
    def patch(activation, hook):
        # Replace the residual at mis_pos with the clean vector
        # activation shape: [batch, seq_pos, d_model]
        activation[0, mis_pos, :] = clean_vec.to(activation.device, activation.dtype)
        return activation

    # Step 3: Generate with the patching hook
    # The hook intercepts the forward pass at layer L and replaces the residual
    with torch.no_grad():
        out = model.generate(
            mis_tokens, max_new_tokens=max_new_tokens, do_sample=False, verbose=False,
            fwd_hooks=[(full_hook, patch)],
        )

    # Return only the NEW tokens (the model's response, not the input prompt)
    return model.to_string(out[0, mis_tokens.shape[1]:])
