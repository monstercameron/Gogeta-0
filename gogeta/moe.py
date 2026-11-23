"""Code-MoE FFN (E60): dropless token-choice top-k mixture of SwiGLU experts.

Replaces a block's dense SwiGLU when ModelConfig.moe_experts > 0 (0 = dense, the default; then this module is never
built and the model is bit-identical to the pre-MoE code, see tests/test_moe.py).

Routing (causal by construction):
  * the router sees only the current token's (normed) hidden state: logits_t = W_r h_t, in fp32;
  * p_t = softmax(logits_t); the top-k experts are chosen per token and their gates renormalised over the k
    (Mixtral style): y_t = sum_{e in topk(t)} (p_te / sum_topk p_t) * SwiGLU_e(h_t);
  * dropless: no capacity factor, no token dropping, no expert-choice routing, so output t never depends on
    any other token (tests/test_moe.py checks this by perturbation).

Auxiliary losses (batch-level losses only, never routing decisions, so they do not break causality):
  * Switch load balance: E * sum_e f_e * P_e with f_e = share of the N*k assignments that went to expert e and
    P_e = mean router probability of e. Exactly 1.0 for uniform routing.
  * router z-loss: mean_t logsumexp(logits_t)^2 (ST-MoE).
The module computes them per call and appends them to the `aux` list it is handed; FusionDance.forward weights
and sums them (see FusionDance.moe_aux).

Experts are batched parameters (w_up [E, d, 2h], w_down [E, h, d]), initialised by FusionDance._init_weights via
reset_parameters(std, std_res) exactly like SwiGLU (up std 0.02, down std_res). The router is an nn.Linear (no
bias), initialised std 0.02 like every Linear; moe_router_frozen=True sets requires_grad=False on it (a frozen
random router).

Dispatch:
  impl "gather" (default): one argsort of the N*k assignments by expert, one gather, a per-expert SwiGLU on its
     contiguous slice, and a deterministic un-permute (index_put with unique indices, no atomics) followed by a sum
     over the k slots. Compute = active params only. One host sync per call (split sizes).
  impl "dense": every expert on every token, masked by the (mostly zero) gate matrix. E/k x the FFN FLOPs, but
     static shapes (torch.compile friendly). Matches "gather" to float tolerance (tests/test_moe.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        E, k, d, h = cfg.moe_experts, cfg.moe_top_k, cfg.d_model, cfg.moe_hidden
        if E < 1 or not 1 <= k <= E:
            raise ValueError(f"need moe_experts >= 1 and 1 <= moe_top_k <= moe_experts (got E={E}, k={k})")
        if cfg.moe_impl not in ("gather", "dense"):
            raise ValueError(f"unknown moe_impl {cfg.moe_impl!r} (gather | dense)")
        self.E, self.k, self.d, self.h = E, k, d, h
        self.impl = cfg.moe_impl
        self.layer_idx = -1  # set by FusionDance (block position), used to group aux terms per layer
        # torch.empty consumes no RNG; values come from reset_parameters (called by FusionDance._init_weights)
        self.w_up = nn.Parameter(torch.empty(E, d, 2 * h))
        self.w_down = nn.Parameter(torch.empty(E, h, d))
        self.router = nn.Linear(d, E, bias=False)
        if cfg.moe_router_frozen:
            self.router.weight.requires_grad_(False)
        # Ternary experts (A3, v0.5.10): BitLinear-style fake quantisation of each expert's weight matrices (absmean
        # ternary, STE) and, with ternary_acts, per-token int8 absmax activations. The router stays full precision.
        self.ternary = bool(getattr(cfg, "ternary", False)) and bool(getattr(cfg, "ternary_ffn", True))
        self.quant_acts = bool(getattr(cfg, "ternary_acts", True))
        self.quant_on = True

    @staticmethod
    def _tq(w: torch.Tensor) -> torch.Tensor:   # absmean ternary per matrix, straight-through
        s = w.abs().mean().clamp(min=1e-5)
        return w + ((w / s).round().clamp(-1, 1) * s - w).detach()

    @staticmethod
    def _aq(x: torch.Tensor) -> torch.Tensor:   # per-token int8 absmax, straight-through
        s = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        return x + ((x * s).round().clamp(-128, 127) / s - x).detach()

    @torch.no_grad()
    def reset_parameters(self, std: float, std_res: float):
        nn.init.normal_(self.w_up, std=std)
        nn.init.normal_(self.w_down, std=std_res)

    def expert_params(self) -> int:
        return self.w_up[0].numel() + self.w_down[0].numel()   # 3 * d * h

    def inactive_params(self) -> int:
        """Expert params not applied to a given token (E - k experts); the router is always active."""
        return (self.E - self.k) * self.expert_params()

    # ------------------------------------------------------------------ routing
    def route(self, x2: torch.Tensor):
        """x2: [N, d] -> (logits fp32 [N,E], probs fp32 [N,E], top-k idx [N,k], renormalised gates fp32 [N,k])."""
        with torch.autocast(device_type=x2.device.type, enabled=False):
            logits = F.linear(x2.float(), self.router.weight.float())
        probs = logits.softmax(dim=-1)
        top_p, top_i = probs.topk(self.k, dim=-1)
        gates = top_p / top_p.sum(dim=-1, keepdim=True)
        return logits, probs, top_i, gates

    def _expert(self, e: int, xe: torch.Tensor) -> torch.Tensor:
        if self.ternary and self.quant_on:
            if self.quant_acts:
                xe = self._aq(xe)
            a, b = torch.matmul(xe, self._tq(self.w_up[e])).chunk(2, dim=-1)
            hdn = F.silu(a) * b
            if self.quant_acts:
                hdn = self._aq(hdn)
            return torch.matmul(hdn, self._tq(self.w_down[e]))
        a, b = torch.matmul(xe, self.w_up[e]).chunk(2, dim=-1)
        return torch.matmul(F.silu(a) * b, self.w_down[e])

    def _gather(self, x2, top_i, gates):
        N, k = top_i.shape
        flat_e = top_i.reshape(-1)                                     # [N*k], slot s = n*k + j
        order = torch.argsort(flat_e, stable=True)
        counts = torch.bincount(flat_e, minlength=self.E).tolist()    # host sync (split sizes)
        xs = x2.index_select(0, order // k)                            # tokens grouped by expert
        ys = [self._expert(e, part) for e, part in enumerate(xs.split(counts)) if part.shape[0] > 0]
        ys = torch.cat(ys, dim=0) if len(ys) > 1 else ys[0]
        ys = ys * gates.reshape(-1)[order].unsqueeze(-1)               # fp32 gates (promotes bf16 under autocast)
        slots = ys.new_empty(N * k, ys.shape[-1])
        slots = slots.index_put((order,), ys)                          # unique indices: deterministic
        return slots.view(N, k, -1).sum(dim=1)

    def _dense(self, x2, top_i, gates):
        G = torch.zeros(x2.shape[0], self.E, dtype=gates.dtype, device=x2.device).scatter(1, top_i, gates)
        y = 0
        for e in range(self.E):
            y = y + self._expert(e, x2) * G[:, e:e + 1]
        return y

    def forward(self, x: torch.Tensor, aux: list | None = None) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        logits, probs, top_i, gates = self.route(x2)
        y = (self._gather if self.impl == "gather" else self._dense)(x2, top_i, gates)
        if aux is not None:
            aux.append(self.stats(logits, probs, top_i))
        return y.reshape(*shape[:-1], y.shape[-1])

    def stats(self, logits, probs, top_i) -> dict:
        N = top_i.shape[0]
        f = torch.bincount(top_i.reshape(-1), minlength=self.E).to(probs.dtype) / (N * self.k)   # assignment share
        P = probs.mean(dim=0)
        return {"layer": self.layer_idx, "grad": torch.is_grad_enabled(),
                "balance": self.E * (f * P).sum(),
                "z": torch.logsumexp(logits, dim=-1).pow(2).mean(),
                "load": f.detach(),
                "entropy": -(probs * probs.clamp_min(1e-20).log()).sum(-1).mean().detach()}


def combine_aux(calls: list, balance_coef: float, z_coef: float) -> dict:
    """Weighted auxiliary loss from the per-call stats of one forward.

    loss = sum over MoE layers of mean over that layer's calls of (balance_coef * balance + z_coef * z).
    A looped core block is called r times per forward; calls made under no_grad (the truncated-BPTT prefix)
    are excluded from the mean whenever the layer has at least one grad-enabled call, so the coefficient acts
    at full strength on the trained calls. Diagnostics are detached: mean balance / z / entropy over layers,
    the max and min expert share over layers, and the per-layer mean load [L, E]."""
    by_layer: dict = {}
    for c in calls:
        by_layer.setdefault(c["layer"], []).append(c)
    loss, bal, z, ent, loads = 0.0, [], [], [], []
    for layer in sorted(by_layer):
        cs = by_layer[layer]
        g = [c for c in cs if c["grad"]] or cs
        b_l = torch.stack([c["balance"] for c in g]).mean()
        z_l = torch.stack([c["z"] for c in g]).mean()
        loss = loss + balance_coef * b_l + z_coef * z_l
        bal.append(b_l.detach())
        z.append(z_l.detach())
        ent.append(torch.stack([c["entropy"] for c in cs]).mean())
        loads.append(torch.stack([c["load"] for c in cs]).mean(0))
    load = torch.stack(loads)
    if not torch.is_tensor(loss):
        loss = load.new_zeros(())
    return {"loss": loss, "balance": torch.stack(bal).mean(), "z": torch.stack(z).mean(),
            "entropy": torch.stack(ent).mean(), "max_load": load.max(), "min_load": load.min(), "load": load}
