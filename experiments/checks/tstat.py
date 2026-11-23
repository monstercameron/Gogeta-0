"""Student t helpers (stdlib only; the WSL venv has no scipy). Two-sided p-values via the regularized incomplete beta,
checked against the df 1 and df 2 closed forms; quantiles by bisection."""
import math

def _betacf(a, b, x, it=300, eps=1e-15):
    qab, qap, qam = a + b, a + 1, a - 1; c = 1.0; d = 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300); h = d
    for m in range(1, it + 1):
        m2 = 2 * m; aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300); c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300); c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300
        de = d * c; h *= de
        if abs(de - 1) < eps: break
    return h
def _betai(a, b, x):  # regularized incomplete beta (Numerical Recipes)
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * _betacf(b, a, 1 - x) / b
def t_two_sided_p(t, df):  # = I_{df/(df+t^2)}(df/2, 1/2); stdlib only (no scipy in the venv)
    return _betai(df / 2, 0.5, df / (df + t * t))


def t_quantile(p_two_sided, df):  # t > 0 with two-sided tail probability p_two_sided
    lo, hi = 0.0, 1e6
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_two_sided_p(mid, df) > p_two_sided: lo = mid
        else: hi = mid
    return (lo + hi) / 2
