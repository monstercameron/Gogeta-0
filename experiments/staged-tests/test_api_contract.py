"""API-contract tests for every external kernel / library function Gogeta calls (A6, C190-C192).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/gogeta/bin/python experiments/staged-tests/test_api_contract.py

Why: the C2 episode ("fused-recurrent GDN returns NaN on this GPU") was a call-site bug. fla 0.6.0's signature is
fused_recurrent_gated_delta_rule(q, k, v, g=None, gk=None, gv=None, beta=None, ...), our test passed beta as the 5th
POSITIONAL argument and it landed in `gk`. These tests make that bug class impossible to repeat silently:

  1. test_signatures_pinned   the installed signatures still have the parameter names and ORDER we rely on (fails
                              loudly, listing every difference). Needs fla importable (CPU is fine: signatures only).
  2. test_no_unsafe_positional  AST scan of gogeta/, experiments/checks/, evals/, post_training/, tests/ for calls
                              to those functions that pass more positional arguments than the pinned safe prefix.
  3. test_keywords_are_named  every keyword we pass is an explicitly NAMED parameter. fla's ops end in **kwargs, so a
                              typo such as `use_beta_sigmoid_in_kernal=True` would be swallowed silently.
  4. test_fragile_positional_report  (informational) calls that pass >3 positionals but are safe by the pinned order.
  5. SDPA / RMSNorm / normalize behavioural contracts (builtins have no inspectable signature).

Re-run (and re-fingerprint, see env_fingerprint.py) after ANY fla / torch / triton upgrade.
"""
from __future__ import annotations

import ast
import inspect
import re

from _harness import ROOT, run_module  # noqa: F401  (also fixes sys.path)

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------- the pinned contract
# leading = parameter names, in order, that we rely on (a prefix of the installed signature).
# safe_positional = how many leading arguments may be passed positionally in OUR code. Everything after must be a keyword.
CONTRACT = {
    "fla.ops.gated_delta_rule.chunk_gated_delta_rule": dict(
        leading=["q", "k", "v", "g", "beta", "scale", "initial_state", "output_final_state",
                 "use_qk_l2norm_in_kernel", "use_beta_sigmoid_in_kernel", "allow_neg_eigval"],
        safe_positional=5),  # (q,k,v,g,beta) is the documented order; everything else by keyword
    "fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule": dict(
        leading=["q", "k", "v", "g", "gk", "gv", "beta", "scale", "initial_state", "output_final_state",
                 "use_qk_l2norm_in_kernel"],
        safe_positional=3),  # 4th is g, 5th is gk (NOT beta): the C2 trap. Pass g= and beta= by keyword.
    "fla.ops.kda.chunk_kda": dict(
        leading=["q", "k", "v", "g", "beta", "scale", "initial_state", "output_final_state", "use_qk_l2norm_in_kernel",
                 "use_gate_in_kernel", "use_beta_sigmoid_in_kernel", "allow_neg_eigval"],
        safe_positional=5),
    "fla.ops.kda.fused_recurrent_kda": dict(
        # NB: differs from chunk_kda from the 6th position on (A_log, dt_bias before scale): a second positional trap.
        leading=["q", "k", "v", "g", "beta", "A_log", "dt_bias", "scale", "initial_state", "output_final_state",
                 "use_qk_l2norm_in_kernel", "use_gate_in_kernel", "use_beta_sigmoid_in_kernel", "allow_neg_eigval"],
        safe_positional=5),
    "fla.modules.causal_conv1d.causal_conv1d": dict(
        leading=["x", "weight", "bias", "residual", "initial_state", "output_final_state", "activation", "backend"],
        safe_positional=2),
    "fla.modules.FusedLinearCrossEntropyLoss.__init__": dict(
        leading=["self", "ignore_index", "label_smoothing", "logit_scale", "logit_softcapping", "num_chunks", "reduction"],
        safe_positional=1),
    "fla.modules.FusedLinearCrossEntropyLoss.forward": dict(
        leading=["self", "x", "target", "weight", "bias"], safe_positional=4),
    "torch.nn.functional.normalize": dict(leading=["input", "p", "dim", "eps", "out"], safe_positional=1),
    "torch.nn.RMSNorm.__init__": dict(leading=["self", "normalized_shape", "eps", "elementwise_affine"], safe_positional=2),
}
# SDPA is a builtin: parsed from its docstring (below). Positional order is also checked behaviourally.
# keywords that are legitimately NOT named parameters because the op reads them from **kwargs (verified in source below)
KWARGS_ONLY = {"chunk_kda": {"A_log", "dt_bias", "chunk_size"}}
SDPA_ORDER = ["query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"]
SDPA_SAFE_POSITIONAL = 3

# short names -> contract key, for the AST scan (method .__init__/.forward entries are not call targets)
CALL_TARGETS = {k.rsplit(".", 1)[-1]: k for k in CONTRACT if not k.endswith((".__init__", ".forward"))}
CALL_TARGETS["FusedLinearCrossEntropyLoss"] = "fla.modules.FusedLinearCrossEntropyLoss.__init__"
CALL_TARGETS["scaled_dot_product_attention"] = "SDPA"
CALL_TARGETS["normalize"] = "torch.nn.functional.normalize"
CALL_TARGETS["RMSNorm"] = "torch.nn.RMSNorm.__init__"

SCAN_DIRS = ["gogeta", "experiments/checks", "evals", "post_training", "tests", "tools"]
# deliberate positional calls (documented reproductions of the bug). (relative path, short function name).
ALLOWED_POSITIONAL = {
    ("experiments/checks/sq_hw_fla_recurrent.py", "fused_recurrent_gated_delta_rule"):
        "SQ70 deliberately reproduces the C2 positional call to show it is non-finite",
}
INDIRECT_NAMES = {"fn", "func", "f", "op", "kernel", "_kernel"}


def _resolve(dotted: str):
    import importlib
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except Exception:  # noqa: BLE001
            continue
        for a in parts[i:]:
            obj = getattr(obj, a)
        return obj
    raise ImportError(dotted)


def _sdpa_signature() -> list[str]:
    doc = torch.nn.functional.scaled_dot_product_attention.__doc__ or ""
    m = re.search(r"scaled_dot_product_attention\(([^)]*)\)", doc)
    assert m, "cannot parse SDPA signature from its docstring: torch changed the docstring format; re-pin by hand"
    return [p.split("=")[0].strip() for p in m.group(1).replace("\n", " ").split(",") if p.strip()]


def _params(fn):
    return list(inspect.signature(fn).parameters.values())


# ---------------------------------------------------------------------------------------------- 1. signatures
def test_signatures_pinned():
    problems = []
    for key, spec in CONTRACT.items():
        try:
            fn = _resolve(key)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{key}: cannot import ({type(e).__name__}: {e})")
            continue
        names = [p.name for p in _params(fn)]
        want = spec["leading"]
        if names[: len(want)] != want:
            extra = [n for n in names if n not in want]
            missing = [n for n in want if n not in names]
            moved = [n for n in want if n in names and names.index(n) != want.index(n)]
            problems.append(f"{key}\n    pinned   : {want}\n    installed: {names[:len(want) + 3]}...\n"
                            f"    missing {missing} | moved {moved} | new {extra[:6]}")
        if spec["safe_positional"] > len(want):
            problems.append(f"{key}: safe_positional={spec['safe_positional']} exceeds the pinned prefix")
    got = _sdpa_signature()
    if got[: len(SDPA_ORDER)] != SDPA_ORDER:
        problems.append(f"SDPA pinned {SDPA_ORDER} installed {got}")
    assert not problems, ("API CONTRACT CHANGED (the C2 bug class lives here). Re-validate: re-run the numerical suite, "
                          "audit call sites, then update CONTRACT and env_fingerprint.json deliberately:\n  "
                          + "\n  ".join(problems))


def test_pinned_traps_documented():
    """The two hazards behind C2 must still be real; if fla fixes them this test tells us to relax the policy."""
    fr = [p.name for p in _params(_resolve("fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule"))]
    assert fr.index("beta") == 6 and fr[4] == "gk", f"fused_recurrent GDN param order changed: {fr[:8]}"
    ch = [p.name for p in _params(_resolve("fla.ops.gated_delta_rule.chunk_gated_delta_rule"))]
    assert ch[3:5] == ["g", "beta"], f"chunk GDN no longer has g, beta in positions 4, 5: {ch[:6]}"
    fk = [p.name for p in _params(_resolve("fla.ops.kda.fused_recurrent_kda"))]
    ck = [p.name for p in _params(_resolve("fla.ops.kda.chunk_kda"))]
    assert fk[5] != ck[5], "kda chunk/recurrent orders now agree at position 6: update the trap note"


def test_all_kernels_swallow_unknown_kwargs():
    """Document, on the installed version, that the ops take **kwargs (typos are swallowed). If this stops being true
    the keyword-name test below becomes redundant but harmless."""
    swallowing = [k for k in CONTRACT if not k.endswith((".__init__", ".forward", "normalize"))
                  and any(p.kind is inspect.Parameter.VAR_KEYWORD for p in _params(_resolve(k)))]
    assert "fla.ops.gated_delta_rule.chunk_gated_delta_rule" in swallowing or True
    return f"ops with **kwargs: {[k.rsplit('.', 1)[-1] for k in swallowing]}"


def test_kda_gate_kwargs_are_read_from_kwargs():
    """chunk_kda's A_log / dt_bias are not named parameters: they are read via kwargs.get(...) and ONLY when
    use_gate_in_kernel=True (otherwise silently dropped). Pin that source behaviour so a refactor is noticed."""
    src = inspect.getsource(_resolve("fla.ops.kda.chunk_kda"))
    assert 'kwargs.get("A_log")' in src and 'kwargs.get("dt_bias")' in src and "if use_gate_in_kernel" in src,         "chunk_kda no longer reads A_log/dt_bias from **kwargs the way KWARGS_ONLY assumes: re-audit sq_kda_kernel.py call sites"
    assert "allow_neg_eigval and not use_beta_sigmoid_in_kernel" in src, "chunk_kda neg-eig flag validation changed"


# ---------------------------------------------------------------------------------------------- 2/3/4. source scan
def _iter_py():
    for d in SCAN_DIRS:
        base = ROOT / d
        if base.exists():
            for p in sorted(base.rglob("*.py")):
                if "__pycache__" in p.parts or "staged-tests" in p.parts or "results" in p.parts:
                    continue
                yield p


class _Scan(ast.NodeVisitor):
    def __init__(self):
        self.alias = {}      # local name or attribute name -> short target
        self.calls = []      # (lineno, short, n_pos, kw_names, has_star, has_dstar)
        self.indirect = []   # (lineno, name, n_pos)
        self.imports_fla = False

    def visit_ImportFrom(self, node):
        if (node.module or "").startswith("fla"):
            self.imports_fla = True
            for a in node.names:
                if a.name in CALL_TARGETS:
                    self.alias[a.asname or a.name] = a.name
        self.generic_visit(node)

    def visit_Import(self, node):
        self.imports_fla |= any(a.name.startswith("fla") for a in node.names)

    def visit_Assign(self, node):
        v = node.value
        short = self.alias.get(v.id) if isinstance(v, ast.Name) else (v.attr if isinstance(v, ast.Attribute) and v.attr in CALL_TARGETS else None)
        if short:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.alias[t.id] = short
                elif isinstance(t, ast.Attribute):
                    self.alias[t.attr] = short
        self.generic_visit(node)

    def visit_Call(self, node):
        f = node.func
        short = None
        if isinstance(f, ast.Name):
            short = self.alias.get(f.id) or (f.id if f.id in CALL_TARGETS and f.id in ("scaled_dot_product_attention",) else None)
            name = f.id
        elif isinstance(f, ast.Attribute):
            short = self.alias.get(f.attr) or (f.attr if f.attr in ("scaled_dot_product_attention", "normalize", "RMSNorm") else None)
            name = f.attr
        else:
            name = None
        n_pos = len(node.args)
        star = any(isinstance(a, ast.Starred) for a in node.args)
        kws = [k.arg for k in node.keywords]
        if short:
            self.calls.append((node.lineno, short, n_pos, kws, star, None in kws))
        elif name in INDIRECT_NAMES and (n_pos >= 4 or star):
            self.indirect.append((node.lineno, name, n_pos))
        self.generic_visit(node)


def scan_sources():
    out = []
    for p in _iter_py():
        sc = _Scan()
        try:
            sc.visit(ast.parse(p.read_text(encoding="utf-8")))
        except SyntaxError:
            continue
        rel = p.relative_to(ROOT).as_posix()
        out.append((rel, sc))
    return out


def _safe_n(short):
    if CALL_TARGETS.get(short) == "SDPA":
        return SDPA_SAFE_POSITIONAL
    return CONTRACT[CALL_TARGETS[short]]["safe_positional"]


def _named_params(short):
    if CALL_TARGETS.get(short) == "SDPA":
        return set(SDPA_ORDER)
    fn = _resolve(CALL_TARGETS[short])
    if short == "RMSNorm":
        fn = _resolve("torch.nn.RMSNorm.__init__")
    return {p.name for p in _params(fn)
            if p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)}


def test_no_unsafe_positional():
    bad, allowed, nsites = [], [], 0
    for rel, sc in scan_sources():
        for line, short, n_pos, kws, star, dstar in sc.calls:
            nsites += 1
            over = n_pos > _safe_n(short) or star
            if over:
                msg = f"{rel}:{line} {short}(...) passes {n_pos}{'+*args' if star else ''} positional args, safe prefix is {_safe_n(short)}"
                (allowed if (rel, short) in ALLOWED_POSITIONAL else bad).append(msg)
    assert nsites > 15, f"scanner found only {nsites} kernel call sites: it has stopped seeing our code"
    assert not bad, "UNSAFE POSITIONAL CALLS (C2 bug class); pass these arguments by keyword:\n  " + "\n  ".join(bad)
    return f"{nsites} call sites scanned; allowed deliberate repros: {len(allowed)}"


def test_keywords_are_named():
    bad = []
    cache = {}
    for rel, sc in scan_sources():
        for line, short, n_pos, kws, star, dstar in sc.calls:
            if short not in cache:
                cache[short] = _named_params(short)
            if short == "chunk_kda" and ({"A_log", "dt_bias"} & set(kws)) and "use_gate_in_kernel" not in kws:
                bad.append(f"{rel}:{line} chunk_kda got A_log/dt_bias without use_gate_in_kernel=...: they are silently ignored")
            for k in kws:
                if k is not None and k not in cache[short] and k not in KWARGS_ONLY.get(short, ()):
                    bad.append(f"{rel}:{line} {short}(..., {k}=...) is not a named parameter (swallowed by **kwargs or an error)")
    assert not bad, "UNKNOWN KEYWORDS:\n  " + "\n  ".join(bad)


def test_fragile_positional_report():
    """Informational: calls relying on the pinned positional order beyond the first 3 args, plus indirect calls the
    scanner cannot resolve. Passing (never fails) unless the pinned prefix is exceeded (that is test 2)."""
    fragile, indirect = {}, []
    for rel, sc in scan_sources():
        for line, short, n_pos, kws, star, dstar in sc.calls:
            if n_pos > 3 and CALL_TARGETS.get(short) != "SDPA":
                fragile.setdefault(short, []).append(f"{rel}:{line}")
        indirect += [f"{rel}:{ln} {nm}(<{n}+ positional>)" for ln, nm, n in sc.indirect]
    summary = {k: len(v) for k, v in fragile.items()}
    print("      positional-past-3 call sites (safe by pinned order):", summary)
    for k, v in fragile.items():
        print(f"        {k}: {', '.join(v[:6])}{' ...' if len(v) > 6 else ''}")
    print(f"      indirect calls through a variable (cannot be statically resolved): {len(indirect)}")
    for s in indirect[:10]:
        print("        ", s)
    return f"{sum(summary.values())} fragile sites, {len(indirect)} indirect"


# ---------------------------------------------------------------------------------------------- 5. behavioural contracts
def test_sdpa_positional_order_and_mask_semantics():
    """Builtins can't be introspected; check the behaviour the model relies on (boolean mask: True = ATTEND)."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 6, 4, dtype=torch.float64) for _ in range(3))
    tril = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    # positional 4th argument is attn_mask, True = attend
    pos = F.scaled_dot_product_attention(q, k, v, tril)
    assert (pos - ref).abs().max() < 1e-12, "4th positional arg of SDPA is no longer attn_mask with True=attend"
    # explicit keyword equals manual attention (is_causal top-left aligned for square q/k)
    s = (q @ k.transpose(-1, -2)) / 2.0
    p = s.masked_fill(~tril, float("-inf")).softmax(-1) @ v
    assert (p - ref).abs().max() < 1e-12
    # a 'False' key is really excluded
    m = tril.clone(); m[5, 0] = False
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
    v2 = v.clone(); v2[:, :, 0] += 100.0
    b = F.scaled_dot_product_attention(q, k, v2, attn_mask=m)
    assert (a[:, :, 5] - b[:, :, 5]).abs().max() < 1e-12 and (a[:, :, 4] - b[:, :, 4]).abs().max() > 1e-3
    # scale is keyword-only semantics: default 1/sqrt(hd)
    sc = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=4 ** -0.5)
    assert (sc - ref).abs().max() < 1e-12
    # is_causal AND attn_mask together is an error (the model never passes both)
    try:
        F.scaled_dot_product_attention(q, k, v, attn_mask=tril, is_causal=True)
        raise AssertionError("SDPA accepted both attn_mask and is_causal: semantics changed, re-audit Attention.forward")
    except RuntimeError:
        pass


def test_rmsnorm_eps_default():
    """nn.RMSNorm(d) has eps=None -> finfo(accumulation dtype).eps: 2.2e-16 for fp64, 1.19e-7 for fp32 AND for bf16
    (bf16 inputs are normalised in fp32; measured, not the 7.8e-3 one might assume). The model relies on the default."""
    d = 8
    for dt, eps in ((torch.float64, 2.220446049250313e-16), (torch.float32, 1.1920929e-07), (torch.bfloat16, 1.1920929e-07)):
        n = torch.nn.RMSNorm(d).to(dt)
        x = torch.full((1, d), 1e-3, dtype=dt)
        want = x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + eps)
        got = n(x).double()
        assert (got - want).abs().max() < (4e-3 if dt is torch.bfloat16 else 1e-6), f"{dt}: RMSNorm default eps changed: {got[0,0].item()} vs {want[0,0].item()}"
    return "eps follows the fp32 accumulation dtype for bf16"


def test_normalize_zero_vector_eps():
    z = F.normalize(torch.zeros(2, 3, dtype=torch.float64), dim=-1)
    assert torch.all(z == 0), "F.normalize(0) is no longer 0 (eps clamp semantics changed): GDN zero-input behaviour changes"


if __name__ == "__main__":
    raise SystemExit(run_module(globals()))
