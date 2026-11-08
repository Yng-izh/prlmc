"""Round 6, module 1: score a slot by how much of the reader's attention it wins.

`disc_leverage` -- the score V10 uses to pick the 360 graft slots -- is a property
of the slot alone: its norm, its dispersion inside its own block, its similarity to
the kept slots. Round 5 showed what that buys at the margin: giving the dropped
blocks 1,008 slots instead of 360 made things monotonically WORSE, so a slot that
looks salient by that score is not a slot the reader can use. Salience is not
readability.

This module measures readability directly, with the frozen reader's own
projections:

    U_i = sum_{l in L} sum_h alpha_i^{(l,h)} * || v_i^{(l,h)} ||_2

`alpha_i^{(l,h)}` is the attention slot `i` receives at layer `l` from head `h`,
computed from the reader's real weights and its real rotary embedding, and
`||v_i||` is the length of what that slot would contribute if it won attention.
A slot the reader never attends to contributes nothing however salient it is; a
slot that attention reaches and that carries a long value vector contributes a
lot. `L` defaults to layers 12, 16, 20, 24, 28 so the score is not a single
layer's opinion.

What the proxy is, exactly
--------------------------
The read path (`methods/memoryllm_modeling/modeling_memoryllm.py`) prepends the
pool to the block and projects ALL of it:

    k_i = W_K * RMSNorm(h_i)      v_i = W_V * RMSNorm(h_i)   (per layer)
    q_t = W_Q * RMSNorm(h_t)      only for the non-memory positions

so a slot's key and value are a function of the stored vector and the layer's
weights, and they do not depend on the query at all. The query hidden states do
not depend on the pool either -- memory positions are dropped from the residual
stream at the end of their layer -- so `h_t` at layer `l` is the hidden state a
plain forward pass of the probe prompt produces. That makes the proxy exact up to
one deliberate approximation, the position it scores at:

    a candidate is scored at the SOURCE position it currently occupies (its index
    in the 19-block window), not at the destination position it will occupy after
    selection. The destination position is a function of the selection being
    computed, so scoring at it would be circular; chronological reordering keeps
    the relative order of the survivors, so the induced ranking is nearly the
    ranking the destination geometry would give.

RoPE is applied rather than dropped: cos/sin come from the reader's own
`rotary_emb` at the positions above, so the score is a real QK product and not a
position-free inner product with an unstated bias. The `--utility-geometry`
escape hatch scores at the compressed geometry instead, for a sensitivity check
that does not need a second implementation.

Whose query
-----------
A fixed bank of TRAINING-side questions, encoded once before the first test row is
read and reused for every row. No test question and no reference answer enters
this module, so the score cannot fit the evaluation set. `--utility-probe-text`
substitutes a single fixed sentence for the bank.

Two questions, one set of projections
------------------------------------
`reading` answers "which slots does the question reach". Round 7 adds `affinity`,
which answers "which slots do the KEPT slots reach" -- the same weights, the same
rotary geometry, with a memory slot in the query position instead of the question.
That substitution is a proxy (the deployed read path never queries with a memory
slot) and the module that uses it has to prove the proxy is worth capacity.

MMR
---
Ranking 2,560 candidates by utility alone returns 684 slots that all score well on
the same kind of content, which is the failure the coverage term exists to prevent.
`mmr_select` charges each pick for its similarity to what is already chosen
(`score = U - penalty * max_sim_to_chosen`), so the budget buys distinct slots
rather than near-duplicates.
"""
import math

import torch
import torch.nn.functional as F

from methods.prlmc.key_safe_projection import effective_projection, read_path_adapter

SCHEMA = "reader_utility/1"
# Five layers, not one: the reader's 32 layers do not agree on what is readable, and
# a single reference layer is exactly the assumption round 5's failure implicates.
UTILITY_LAYERS = (12, 16, 20, 24, 28)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rope(tensor, cos, sin):
    """`[T, H, Dh]` rotated with `cos`/`sin` of `[T, Dh]` (reader's own layout)."""
    return tensor * cos.unsqueeze(1) + _rotate_half(tensor) * sin.unsqueeze(1)


class SlotReading:
    """What the reader can see of a set of slots, at the utility layers.

    `key` and `value` are `[N, layers * kv_heads * head_dim]` unit rows: each
    layer's vector is normalised on its own before the layers are concatenated, so
    no layer dominates the similarity because its key happens to be longer.
    """

    def __init__(self, layers, key, value, utility, head_dim):
        self.layers = tuple(layers)
        self.key = key
        self.value = value
        self.utility = utility
        self.head_dim = int(head_dim)

    @property
    def n_slots(self):
        return int(self.utility.numel())

    def summary(self):
        u = self.utility
        if u is None:
            return dict(slots=int(self.key.shape[0]), layers=list(self.layers),
                        utility=None, key_dim=int(self.key.shape[-1]),
                        value_dim=int(self.value.shape[-1]))
        return dict(slots=int(u.numel()), layers=list(self.layers),
                    utility_mean=float(u.mean()), utility_std=float(u.std()),
                    utility_min=float(u.min()), utility_max=float(u.max()),
                    key_dim=int(self.key.shape[-1]),
                    value_dim=int(self.value.shape[-1]))


class ReaderScorer:
    """The frozen reader's reading of a slot, computed once and reused.

    Holds, per utility layer, the *effective* query/key/value maps (the decoder
    LoRA delta folded in -- `key_safe_projection.effective_projection`, which
    `build_key_bases` already checks against the real modules) and the reader's own
    rotary embedding. The norm is the reader's own `input_layernorm`, called on
    float32 input, so the eps and the cast order are the deployed ones.
    """

    def __init__(self, reader, layers=UTILITY_LAYERS):
        self.reader = reader
        self.adapter = read_path_adapter(reader)
        config = reader.config
        self.hidden = int(config.hidden_size)
        self.heads = int(config.num_attention_heads)
        self.kv_heads = int(config.num_key_value_heads)
        if self.heads % self.kv_heads:
            raise ValueError(f"{self.heads} query heads do not group into "
                             f"{self.kv_heads} key/value heads")
        self.groups = self.heads // self.kv_heads
        self.head_dim = self.hidden // self.heads
        self.layers = tuple(int(index) for index in layers)
        depth = len(list(reader.model.layers))
        for index in self.layers:
            if not 0 <= index < depth:
                raise ValueError(f"utility layer {index} outside a {depth}-layer reader")
        device = reader.memory.device
        self.device = device
        self.maps = {}
        for index in self.layers:
            layer = reader.model.layers[index]
            self.maps[index] = dict(
                q=effective_projection(layer.self_attn.q_proj, self.adapter).to(device),
                k=effective_projection(layer.self_attn.k_proj, self.adapter).to(device),
                v=effective_projection(layer.self_attn.v_proj, self.adapter).to(device),
                norm=layer.input_layernorm,
                rotary=layer.self_attn.rotary_emb,
            )

    # -- the pieces ---------------------------------------------------------
    def _cos_sin(self, entry, positions):
        """`[T, Dh]` cos/sin for `positions`, from the reader's own rotary module."""
        reference = entry["k"]
        dummy = reference.new_zeros(1, len(positions), self.hidden)
        cos, sin = entry["rotary"](dummy, positions.unsqueeze(0))
        return cos[0], sin[0]

    @torch.no_grad()
    def probe_bank(self, probe_ids, keep_last=64):
        """Query hidden states of the probe prompt at every utility layer.

        The pool is not present in this forward and does not need to be: memory
        positions never reach the residual stream, so the query's hidden state at
        layer `l` is the same with and without a window installed. `keep_last`
        bounds the tokens that are allowed to speak for the question (the prompt is
        `question + instruction`, so the tail carries the question).
        """
        ids = torch.as_tensor([list(probe_ids)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.reader(input_ids=ids, use_cache=False, return_dict=True,
                              output_hidden_states=True)
        hidden = out.hidden_states
        states = {}
        for index in self.layers:
            # `hidden[l]` is the INPUT to layer l, which is what the layer's
            # attention normalises; `hidden[l+1]` is its output.
            states[index] = hidden[index].float()
        del out, hidden
        if keep_last and states[self.layers[0]].shape[1] > keep_last:
            states = {index: value[:, -keep_last:] for index, value in states.items()}
        return states

    @torch.no_grad()
    def reading(self, slots, positions, probes=None, query_offset=None):
        """`slots` `[all_layers, N, D]` -> `SlotReading` at the utility layers.

        `positions` is the source position of each slot (`[N]`); `query_offset` is
        where the question sits relative to the slots (default: after all of them),
        which is the causal geometry the reader actually has.

        `probes=None` skips the question side entirely: the key and value
        representations are still built with the reader's own projections and rotary
        embedding, but nothing is scored against a question and `utility` comes back
        as None. That path exists so a selector can use the reader's read path while
        keeping every external question out of the decision.
        """
        n_slots = int(slots.shape[1])
        if positions.numel() != n_slots:
            raise ValueError(f"{positions.numel()} positions for {n_slots} slots")
        probes = list(probes or [])
        offset = int(n_slots if query_offset is None else query_offset)
        key_parts, value_parts = [], []
        utility = (torch.zeros(n_slots, device=self.device, dtype=torch.float32)
                   if probes else None)
        per_layer = []
        for index in self.layers:
            entry = self.maps[index]
            stored = slots[index].to(self.device)
            normed = entry["norm"](stored.float())
            keys = F.linear(normed, entry["k"]).view(n_slots, self.kv_heads, self.head_dim)
            values = F.linear(normed, entry["v"]).view(n_slots, self.kv_heads,
                                                       self.head_dim)
            cos_k, sin_k = self._cos_sin(entry, positions.to(self.device))
            keys = _rope(keys, cos_k, sin_k)
            value_norm = values.float().norm(dim=-1)
            layer_utility = torch.zeros(n_slots, device=self.device, dtype=torch.float32)
            scale = math.sqrt(self.head_dim)
            for hidden in probes:
                states = hidden[index]
                tokens = states.shape[1]
                q_positions = offset + torch.arange(tokens, device=self.device)
                query = entry["norm"](states[0].float())
                q = F.linear(query, entry["q"]).view(tokens, self.heads, self.head_dim)
                cos_q, sin_q = self._cos_sin(entry, q_positions)
                q = _rope(q, cos_q, sin_q).view(tokens, self.kv_heads, self.groups,
                                                self.head_dim)
                logits = torch.einsum("tkgd,nkd->tnkg", q, keys) / scale
                attention = torch.softmax(logits, dim=1).mean(dim=0)
                layer_utility += (attention * value_norm.unsqueeze(-1)).sum(dim=(1, 2))
                del logits, attention, q
            if probes:
                utility += layer_utility
            # Per layer, unit-normalised, then the layers are concatenated and the
            # concatenation re-normalised: equal say per layer.
            key_parts.append(F.normalize(keys.reshape(n_slots, -1).float(), dim=-1))
            value_parts.append(F.normalize(values.reshape(n_slots, -1).float(), dim=-1))
            record = dict(layer=index,
                          key_rms=float(keys.float().pow(2).mean().sqrt()),
                          value_rms=float(values.float().pow(2).mean().sqrt()))
            if probes:
                record.update(utility_mean=float(layer_utility.mean()),
                              utility_max=float(layer_utility.max()))
            per_layer.append(record)
            del keys, values, normed
        key = F.normalize(torch.cat(key_parts, dim=-1), dim=-1)
        value = F.normalize(torch.cat(value_parts, dim=-1), dim=-1)
        return SlotReading(self.layers, key, value, utility, self.head_dim), per_layer

    @torch.no_grad()
    def affinity(self, slots, positions, queries, query_positions, query_chunk=32):
        """Attention mass the given native QUERY slots place on each candidate slot.

        `reading` asks what the question's hidden states reach. This asks the other
        question the window needs answered: from the point of view of slots that are
        already kept, which slots elsewhere in the window belong with them? Both
        sides go through the reader's own projections at the SOURCE positions they
        occupy, so this is a real QK product under the reader's own rotary geometry.

        It is a PROXY, and deliberately a weak one: the reader never projects a
        memory slot into a query -- in the deployed read path `q` comes from the
        question only. A memory slot acting as a query is a way to ask "which
        candidates would this kept slot look at", not a measurement of what the
        reader will do with them. Whether the proxy is worth its capacity is exactly
        what the arm that uses it has to prove.

        Pooling: per query seed, per layer, the softmax runs over the candidate pool,
        and the MAXIMUM over that seed's query-group/head pairs is kept before the
        seeds are averaged. Mean-pooling over heads was tried first and is unusable at
        this pool size -- 8 kv heads x 4 query groups over ~3.6k candidates averages
        to noise around 1/N.

        Returns `(mass [N], per_layer)`; `mass` is not normalised, its scale is
        "attention share averaged over seeds", and callers min-max it before use.
        """
        n_slots = int(slots.shape[1])
        n_queries = int(queries.shape[1])
        if positions.numel() != n_slots:
            raise ValueError(f"{positions.numel()} positions for {n_slots} candidates")
        if query_positions.numel() != n_queries:
            raise ValueError(f"{query_positions.numel()} positions for {n_queries} "
                             f"queries")
        if n_slots == 0 or n_queries == 0:
            raise ValueError("affinity needs at least one candidate and one query")
        mass = torch.zeros(n_slots, device=self.device, dtype=torch.float32)
        per_layer = []
        scale = math.sqrt(self.head_dim)
        for index in self.layers:
            entry = self.maps[index]
            stored = slots[index].to(self.device)
            normed = entry["norm"](stored.float())
            keys = F.linear(normed, entry["k"]).view(n_slots, self.kv_heads, self.head_dim)
            cos_k, sin_k = self._cos_sin(entry, positions.to(self.device))
            keys = _rope(keys, cos_k, sin_k)
            asked = queries[index].to(self.device)
            asked_normed = entry["norm"](asked.float())
            q = F.linear(asked_normed, entry["q"]).view(n_queries, self.heads,
                                                        self.head_dim)
            cos_q, sin_q = self._cos_sin(entry, query_positions.to(self.device))
            q = _rope(q, cos_q, sin_q).view(n_queries, self.kv_heads, self.groups,
                                            self.head_dim)
            layer_mass = torch.zeros(n_slots, device=self.device, dtype=torch.float32)
            for start in range(0, n_queries, query_chunk):
                logits = torch.einsum("mkqd,nkd->mnkq", q[start:start + query_chunk],
                                      keys) / scale
                attention = torch.softmax(logits, dim=1)
                layer_mass += attention.amax(dim=(2, 3)).sum(dim=0)
                del logits, attention
            layer_mass /= float(n_queries)
            mass += layer_mass
            per_layer.append(dict(layer=index, mass_mean=float(layer_mass.mean()),
                                  mass_max=float(layer_mass.max()),
                                  mass_std=float(layer_mass.std())))
            del keys, q, normed, asked_normed
        return mass, per_layer


@torch.no_grad()
def mmr_select(utility, key, wanted, penalty=0.5, chunk=128):
    """Greedy maximum-marginal-relevance pick: utility minus duplicate cost.

    Returns indices into the candidate arrays. `key` must be unit rows (it is, from
    `SlotReading`). The first pick is the highest utility; every later pick pays
    `penalty` for its similarity to the closest slot already chosen, so a run of
    near-identical slots stops after its first member.
    """
    n = int(utility.numel())
    if wanted > n:
        raise ValueError(f"asked for {wanted} of {n} candidates")
    if key.shape[0] != n:
        raise ValueError("utility and key disagree on the candidate count")
    scale = float(utility.max() - utility.min())
    base = ((utility - utility.min()) / max(scale, 1e-12)).float()
    chosen = torch.empty(wanted, dtype=torch.long, device=utility.device)
    best = torch.zeros(n, device=utility.device, dtype=torch.float32)
    taken = torch.zeros(n, dtype=torch.bool, device=utility.device)
    score = base.clone()
    for step in range(wanted):
        index = int(torch.argmax(score))
        chosen[step] = index
        taken[index] = True
        if step + 1 < wanted:
            similarity = key[index] @ key.t()
            best = torch.maximum(best, similarity)
            score = base - penalty * best
            score[taken] = -torch.inf
    return chosen


def probe_report(probes, scorer, sources):
    return dict(schema=SCHEMA, layers=list(scorer.layers), probes=len(probes),
                heads=scorer.heads, kv_heads=scorer.kv_heads,
                head_dim=scorer.head_dim, adapter=scorer.adapter,
                geometry="source-window positions; the question is placed after the "
                         "whole window",
                probe_sources=sources,
                note="no test question and no reference answer enters this score")
