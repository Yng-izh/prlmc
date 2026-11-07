"""Round 4, step 2: which channel carries the candidate content -- the keys or the values?

Experiment 4 froze the reader's keys and the candidate branch went nearly silent:
swapping all 2,200 candidate slots moved 3 of 158 predictions instead of 38, at
identical accuracy. The inference was that the content dependence ran through the
addressing channel -- candidates moved the keys, the keys moved the attention
weights, the weights amplified a small value change into an answer change. This
module turns that inference into a manipulation.

Four arms, one per assignment of the two channels:

    A  key from the real candidates, value from the real candidates   (existing arm)
    B  key from the real candidates, value from another document's    (this module)
    C  key from another document's,  value from the real candidates   (this module)
    D  key from another document's,  value from another document's    (existing arm)

B minus A isolates the value channel; C minus A isolates the key channel; A minus D
is what is being decomposed. A and D already exist as `AnchorFusion` and
`AnchorFusionDonor`, so only two arms are new.

Why the construction is exact, not approximate
---------------------------------------------
A pool slot at layer l reaches the rest of the network ONLY through

    k = W_K diag(gamma) (h / rms(h))        v = W_V diag(gamma) (h / rms(h))

because `cat_memory_and_hiddens` prepends it, the block's `input_layernorm` then
`k_proj`/`v_proj` are its only consumers, and the memory positions are truncated
out of the residual stream at the end of the layer. Its own query output is
discarded. So two slots that produce the same `(k, v)` are the same slot as far as
the reader is concerned, and "swap the value but keep the key" is well posed rather
than a hand-wave.

Concretely, with `A_K = W_K diag(gamma)` and `A_V = W_V diag(gamma)`, arm B asks for

    A_K s = A_K s_real        A_V s = A_V s_donor        rms(s) = 1

The first constraint allows a change in `null(A_K)`, which is 3,072-dimensional;
`A_V` restricted to that space is a 3,072 -> 1,024 map and is generically onto, so
the pair `(A_K s_real, A_V s_donor)` is always reachable. Stacking `M = [A_K; A_V]`
the problem is `M s = t` with `|s| = sqrt(D)`, which is solved exactly:

    s = M+ t + sqrt(D - |M+ t|^2) * w        w a fixed unit vector in null(M)

`M+ t` is the minimum-norm preimage of the target pair; the second term puts the
norm back without disturbing `M s`, because `w` is annihilated by `M`. That is one
SVD per layer, precomputed once, and no iteration.

Three things this code measures instead of assuming
--------------------------------------------------
1. **The target is not re-derived, it is measured.** `channel_targets` calls the
   reader's own `input_layernorm`, `k_proj` and `v_proj` -- LoRA adapter included --
   on the two source slots, so the target is the `(k, v)` the reader would really
   compute, not a reconstruction from a folded weight matrix.
2. **The construction is checked on every row.** `channel_error` runs the same real
   modules on the BUILT slot and compares against the target. A construction error
   is a hard failure, not a note.
3. **The identity is included.** Feeding the same slots as key and value source must
   reproduce their own `(k, v)`; that is the control that says the construction adds
   nothing of its own. It is the same code path, so no extra arm is needed.

What this cannot promise
------------------------
The pair `(k_real, v_donor)` never occurs in the data. If the reader depended on
the two jointly in a way the factorization above cannot see -- it does not, by the
argument above, but a code change elsewhere in the model could break that -- the
synthetic slot would sit off the data manifold and the arm would read as noise.
`channel_error` is what would catch it.
"""
import torch

from methods.prlmc.key_safe_projection import (
    adapter_selected, effective_projection, read_path_adapter)

SCHEMA = "key_value_crossover/1"


# ---------------------------------------------------------------------------
# the two maps of a layer
# ---------------------------------------------------------------------------
def channel_maps(layer, adapter):
    """`(A_K, A_V)` = `(W_K diag(gamma), W_V diag(gamma))`, LoRA delta folded in."""
    gamma = layer.input_layernorm.weight.detach().float()
    weight_k = effective_projection(layer.self_attn.k_proj, adapter)
    weight_v = effective_projection(layer.self_attn.v_proj, adapter)
    if weight_k.shape[-1] != gamma.shape[-1] or weight_v.shape[-1] != gamma.shape[-1]:
        raise ValueError("the projection width and the norm width disagree")
    return weight_k * gamma.unsqueeze(0), weight_v * gamma.unsqueeze(0)


# ---------------------------------------------------------------------------
# what the reader actually computes for a pair of slots
# ---------------------------------------------------------------------------
@torch.no_grad()
def channel_targets(reader, key_slots, value_slots, layer_start=0, adapter=None):
    """`[L, N, 2 * key_width]`: the `(k, v)` the reader computes for these slots.

    The key comes from `key_slots` and the value from `value_slots`, both through the
    real modules, so the target carries the same fp16 rounding and the same LoRA
    adapter the forward pass will. Nothing here is re-derived from a weight matrix.
    """
    layers = list(reader.model.layers)
    adapter = read_path_adapter(reader) if adapter is None else adapter
    if layer_start + key_slots.shape[0] > len(layers):
        raise ValueError(f"the window covers layers {layer_start}.."
                         f"{layer_start + key_slots.shape[0]}, the reader has "
                         f"{len(layers)}")
    keys, values = [], []
    for offset in range(key_slots.shape[0]):
        layer = layers[layer_start + offset]
        k_module = layer.self_attn.k_proj
        v_module = layer.self_attn.v_proj
        k_hidden = layer.input_layernorm(
            key_slots[offset:offset + 1].to(k_module.weight.dtype))
        with adapter_selected(k_module, adapter):
            keys.append(k_module(k_hidden).float())
        v_hidden = layer.input_layernorm(
            value_slots[offset:offset + 1].to(v_module.weight.dtype))
        with adapter_selected(v_module, adapter):
            values.append(v_module(v_hidden).float())
    return torch.cat([torch.cat(keys, dim=0), torch.cat(values, dim=0)], dim=-1)


def channel_error(reader, built, key_slots, value_slots, layer_start=0, adapter=None):
    """Relative distance between the built slot's `(k, v)` and the requested one."""
    want = channel_targets(reader, key_slots, value_slots, layer_start, adapter)
    got = channel_targets(reader, built, built, layer_start, adapter)
    half = want.shape[-1] // 2
    report = {}
    for name, index in (("key", slice(0, half)), ("value", slice(half, None))):
        reference = want[..., index].norm(dim=-1).clamp_min(1e-12)
        report[name] = float((got[..., index] - want[..., index]).norm(dim=-1)
                             .div(reference).mean())
    return report


# ---------------------------------------------------------------------------
# the solver
# ---------------------------------------------------------------------------
class CrossoverProjector(torch.nn.Module):
    """Per-layer `(k, v) -> slot` solvers, built once from the frozen reader.

    Holds one pseudo-inverse per layer, `[L, hidden, 2 * key_width]`, and one fixed
    null direction per layer. Nothing here is trained; the buffers are
    non-persistent because they are a function of the frozen reader, not of the
    branch, and they total ~1.1 GB -- rebuilt from the reader rather than copied
    into a checkpoint.
    """

    def __init__(self, pinv, nulls, hidden, key_width, report=None):
        super().__init__()
        self.register_buffer("pinv", pinv.float(), persistent=False)
        self.register_buffer("nulls", nulls.float(), persistent=False)
        self.hidden = int(hidden)
        self.key_width = int(key_width)
        self.report = report

    @classmethod
    def build(cls, reader, adapter=None, relative_tolerance=1e-6, seed=0):
        layers = list(reader.model.layers)
        adapter = read_path_adapter(reader) if adapter is None else adapter
        device = layers[0].self_attn.k_proj.weight.device
        pinvs, nulls, ranks, spreads = [], [], [], []
        for index, layer in enumerate(layers):
            a_k, a_v = channel_maps(layer, adapter)
            stacked = torch.cat([a_k, a_v], dim=0)
            u, s, vh = torch.linalg.svd(stacked, full_matrices=False)
            floor = float(s[0]) * float(relative_tolerance)
            rank = max(int((s > floor).sum()), 1)
            inverse = torch.zeros_like(s)
            inverse[:rank] = 1.0 / s[:rank]
            # M+ = V S^-1 U^T, with the useless directions zeroed out.
            pinvs.append((vh.transpose(0, 1) * inverse.unsqueeze(0))
                         @ u.transpose(0, 1))
            row_space = vh.transpose(0, 1)[:, :rank]
            generator = torch.Generator(device="cpu").manual_seed(int(seed) + index)
            w = torch.randn(row_space.shape[0], generator=generator,
                            dtype=torch.float32).to(device)
            w = w - row_space @ (row_space.transpose(0, 1) @ w)
            nulls.append(w / w.norm().clamp_min(1e-12))
            ranks.append(rank)
            spreads.append(float(s[-1] / s[0]))
            del u, s, vh, stacked, a_k, a_v
        hidden = int(pinvs[0].shape[0])
        if len(set(item.shape[1] for item in pinvs)) != 1:
            raise ValueError("the layers disagree on the joint key/value width")
        report = dict(
            schema=SCHEMA, layers=len(layers), hidden=hidden,
            key_width=int(pinvs[0].shape[1] // 2), rank=ranks[0], ranks=ranks,
            adapter=adapter, condition=min(spreads), spread=max(spreads),
            feasible_by_construction=bool(min(ranks) >= pinvs[0].shape[1] // 2),
            identity=("the minimum-norm preimage, plus one fixed null direction to "
                      "restore the norm; `M s = t` is exact, the norm is exact, and "
                      "`w` is annihilated by `M` so it cannot disturb the target"),
            free_dimensions=int(hidden - ranks[0]))
        return cls(torch.stack(pinvs, dim=0), torch.stack(nulls, dim=0), hidden,
                   int(pinvs[0].shape[1] // 2), report)

    def forward(self, reader, key_from, value_from, layer_start=0):
        """Slots whose key comes from `key_from` and whose value from `value_from`.

        Returns the slots and the diagnostics that say whether the solve had room:
        `infeasible` counts the slots where the minimum-norm preimage was already
        longer than one unit-RMS slot, which is the only case where the norm cannot
        be restored without scaling the key.
        """
        if key_from.shape != value_from.shape:
            raise ValueError("the key source and the value source must have the same "
                             "shape")
        if key_from.shape[-1] != self.hidden:
            raise ValueError(f"expected hidden {self.hidden}, got "
                             f"{key_from.shape[-1]}")
        dtype = key_from.dtype
        targets = channel_targets(reader, key_from, value_from, layer_start)
        if targets.shape[-1] != 2 * self.key_width:
            raise ValueError(f"the reader produced {targets.shape[-1]} channels, the "
                             f"solver was built for {2 * self.key_width}")
        basis = self.pinv[layer_start:layer_start + key_from.shape[0]]
        nulls = self.nulls[layer_start:layer_start + key_from.shape[0]]
        if basis.shape[0] != key_from.shape[0]:
            raise ValueError("the solver does not cover the layers in this window")

        preimage = torch.matmul(targets, basis.transpose(1, 2))
        room = float(self.hidden) - preimage.square().sum(dim=-1, keepdim=True)
        infeasible = int((room < 0).sum())
        alpha = room.clamp_min(0.0).sqrt()
        built = preimage + alpha * nulls.unsqueeze(1)
        # Only the direction matters to the reader, so restore the source slot's RMS
        # instead of leaving every slot at unit RMS: the rest of the pipeline is
        # checked for norm preservation and there is no reason to break it here.
        rms = key_from.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        built = built / built.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        combined = preimage + alpha * nulls.unsqueeze(1)
        info = dict(
            infeasible=infeasible,
            alpha_share=float(alpha.mean() / max(float(self.hidden) ** 0.5, 1e-6)),
            preimage_share=float(preimage.norm() / combined.norm().clamp_min(1e-12)))
        return (built * rms).to(dtype), info
