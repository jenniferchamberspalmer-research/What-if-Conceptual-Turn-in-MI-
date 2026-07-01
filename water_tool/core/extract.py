"""Canonical, single-source residual extraction for Gemma 2 2B.

Both experiments in this repository read residual state through THIS module,
so extraction cannot drift between them:

  - The Water Pattern Tool (View 3 / SAE) reads the residual OUTPUT of a
    single block via `residual_at_layer`.
  - The Dichotomy Transformation Probe reads the full 0->output layer sweep
    via `residual_sweep`.

Layer convention (Gemma 2 2B has 26 decoder blocks):

    output_hidden_states returns a tuple of length 27.
        index 0   -> the embedding output (the INPUT residual state, "layer 0")
        index i   -> the residual OUTPUT of block i-1   (1 <= i <= 26)
    So the residual output of block `layer` is hidden_states[layer + 1],
    and the full sweep read across "layers 0 to output" is hidden_states[0..26].

`residual_at_layer(text, layer)` returns exactly the tensor the previous
forward-hook implementation returned (the residual output of block `layer`).
A numeric self-check, `hook_matches_hidden_states`, confirms the two paths
agree so the single-source refactor is provably behaviour-preserving.
"""

import torch

from .model import load


def find_target_position(text: str, target: str, tok) -> int:
    """Return the token index whose character range covers the end of `target`.

    Uses the fast tokenizer's offset_mapping. add_special_tokens=True keeps
    the BOS token in the offsets list, so the returned index is already
    aligned with the model's forward pass.
    """
    char_pos = text.find(target)
    if char_pos < 0:
        raise ValueError(f"Target word '{target}' not found in sentence.")
    char_end = char_pos + len(target)

    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    for i, (start, end) in enumerate(offsets):
        if start < char_end and end >= char_end:
            return i
    return len(enc["input_ids"]) - 1


@torch.no_grad()
def residual_sweep(text: str):
    """One forward pass; return the residual state at every layer 0..output.

    Returns (hidden_states, enc) where
        hidden_states : Tensor [n_layers + 1, seq, hidden]  (27 for Gemma 2 2B)
        enc           : the tokenizer BatchEncoding used for the pass.

    hidden_states[0] is the embedding (input) state; hidden_states[i] is the
    residual output of block i-1. This is the layer-sweep harness shared by
    both experiments.
    """
    model, tok = load()
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model(**enc, output_hidden_states=True)
    # Stack the tuple into one tensor and drop the batch dim.
    hs = torch.stack(out.hidden_states, dim=0)[:, 0, :, :]  # [n_layers+1, seq, hidden]
    return hs.detach(), enc


@torch.no_grad()
def residual_at_layer(text: str, layer: int):
    """Residual OUTPUT of block `layer`. Returns (residual_seq [seq, hidden], enc).

    Behaviour-preserving replacement for the old forward-hook capture used by
    View 3. Equals hidden_states[layer + 1] from `residual_sweep`.
    """
    hs, enc = residual_sweep(text)
    return hs[layer + 1], enc


@torch.no_grad()
def hook_matches_hidden_states(text: str, layer: int) -> bool:
    """Self-check: the forward-hook capture equals hidden_states[layer + 1].

    Used once during the single-source refactor to prove that routing the SAE
    view through `residual_at_layer` does not change its inputs.
    """
    model, tok = load()
    captured = {}

    def hook(_module, _input, output):
        x = output[0] if isinstance(output, tuple) else output
        captured["x"] = x

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        enc = tok(text, return_tensors="pt").to(model.device)
        model(**enc)
    finally:
        handle.remove()

    hook_out = captured["x"][0]
    sweep_out, _ = residual_at_layer(text, layer)
    return torch.allclose(hook_out.float(), sweep_out.float(), atol=1e-4)
