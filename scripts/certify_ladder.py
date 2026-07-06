#!/usr/bin/env python3
"""
CERTIFY (ladder, K-agnostic): exact certified weak-duality upper bound on S*(K).
Derived verbatim from certify_sstar.py; the ONLY changes are (a) K/M/TOL are read
from the npz so the exact residual pass covers ALL k=2..K for any K, and (b) the
sanity block is parameterized (LP primal read from --report; monotone floor =
banked S*(4800); ceiling sanity 1.0002).

Bound (weak duality, VALID for ANY y >= 0):
   B(y) = TOL * sum_m y_m  +  10 * sum_{k=2}^{K} |r_k|,   B(y) >= S*(K)
   r_k = c_k - sum_m y_m a_{m,k},  c_k = -ln(k)/k,  a_{m,k} = floor(m/k) - m/k.

CERTIFIED ARITHMETIC (no float on the bound path):
 * y_m = Y_m / D exact (D = 2^denom_pow).
 * a_{m,k} = (k*floor(m/k) - m)/k exact; k*floor(m/k)-m = -(m mod k) exact int.
 * q_k = sum_m y_m a_{m,k} = N_k/(k*D), N_k = sum_m Y_m*(k*floor(m/k)-m) big-int.
 * |r_k| = |ln(k) + N_k/D| / k; ln(k) via outward-rounded mp.iv.log; whole bound
   assembled in mp.iv interval arithmetic; output UPPER endpoint => certified.
OVERFLOW note: per-term product Y_m*(k*floor(m/k)-m) is done in Python big-ints
(object dtype), so it never overflows regardless of K; the int64 assertion is a
belt-and-braces gate.
"""
import sys, json, hashlib, time, argparse
import numpy as np
from fractions import Fraction
from mpmath import mp, iv

# Cross-reach monotonicity of S*(K) is measured in all solves; not proven; not
# used in the certificate. This string is the ONLY monotonicity claim emitted to
# the machine-readable artifacts (verbatim, so a reader cannot mistake it for a
# proven lemma).
MONOTONE_NOTE = "Measured in all solves; not proven; not used in the certificate."

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("out", nargs="?", default="certificate.json")
ap.add_argument("--report", default="", help="report_K{K}.json for LP-primal sanity")
ap.add_argument("--prec", type=int, default=400)
ap.add_argument("--ceiling-sanity", type=float, default=1.0002)
args = ap.parse_args()

mp.prec = args.prec
iv.prec = args.prec

d = np.load(args.npz)
m_arr = d["m"].astype(np.int64)
Y_arr = d["Y"].astype(np.int64)
denom_pow = int(d["denom_pow"])
K = int(d["K"]); M = int(d["M"]); TOL = float(d["TOL"])
D = 1 << denom_pow

# The verifier samples x in [1, 10K), so floor(x) <= 10K-1: the integer
# constraint index m ranges over [1, 10K-1]. The npz field M stores the
# verifier's sampling upper_bound (10K); the constraint-domain upper is M_dom.
assert M == 10 * K, f"npz M ({M}) must equal 10*K ({10*K}) (verifier upper_bound)"
M_dom = 10 * K - 1                       # true constraint-domain upper (m <= 10K-1)
assert TOL == 1.0001

# ---- (1) y -> exact dyadic; clip negatives to 0 (preserves cert validity) ---
neg = int((Y_arr < 0).sum())
if neg:
    print(f"[clip] {neg} negative Y clipped to 0")
    Y_arr = np.where(Y_arr < 0, 0, Y_arr)
assert (Y_arr >= 0).all()
assert m_arr.min() >= 1 and m_arr.max() <= M_dom, (int(m_arr.min()), int(m_arr.max()), M_dom)
maxYval = int(Y_arr.max())
assert maxYval * (K - 1) < (1 << 63), "int64 overflow risk in per-term product"

S_Y = int(Y_arr.sum())
sum_y_exact = Fraction(S_Y, D)
print(f"K={K} M={M} rows={len(m_arr)} D=2^{denom_pow} sum_y={float(sum_y_exact):.15g}")

# ---- (2) exact integer residual pass over ALL k = 2..K ----------------------
Y_obj = Y_arr.astype(object)
t0 = time.time()
iv_D = iv.mpf(D)
sum_r_iv = iv.mpf(0)
sum_abs_r_float = 0.0
max_abs_r_float = 0.0
report_interval = max(500, K // 24)
for k in range(2, K + 1):
    delta = k * (m_arr // k) - m_arr                 # int64 vector, exact
    Nk = int((Y_obj * delta).sum())                  # big-int exact
    lnk = iv.log(k)
    Ak = iv.mpf(Nk) / iv_D
    val = lnk + Ak
    ub = iv.mpf(max(abs(val.a), abs(val.b)))
    abs_r = ub / iv.mpf(k)
    sum_r_iv = sum_r_iv + abs_r
    fr = (float(lnk.b) + Nk / D) / k
    a_r = abs(fr)
    sum_abs_r_float += a_r
    if a_r > max_abs_r_float:
        max_abs_r_float = a_r
    if k % report_interval == 0:
        print(f"  k={k}  sum|r|~{sum_abs_r_float:.6e}  max|r|~{max_abs_r_float:.3e}  t={time.time()-t0:.1f}s")

print(f"[residual pass] done in {time.time()-t0:.1f}s  "
      f"float sum|r|={sum_abs_r_float:.6e} max|r|={max_abs_r_float:.3e}")

# ---- (3)/(4) assemble B with outward rounding -------------------------------
term1_iv = iv.mpf(10001 * S_Y) / iv.mpf(10000 * D)
term2_iv = iv.mpf(10) * sum_r_iv
B_iv = term1_iv + term2_iv
B_upper = B_iv.b
sum_r_upper = sum_r_iv.b
term1_upper = term1_iv.b

mp.prec = args.prec
B_up_mpf = mp.mpf(B_upper)

def decimal_ceiling(x_mpf, ndig):
    import mpmath as _m
    s = _m.nstr(x_mpf, ndig + 3, strip_zeros=False)
    from decimal import Decimal, ROUND_CEILING, getcontext
    getcontext().prec = ndig + 5
    dv = Decimal(s)
    adj = dv.adjusted()
    quant_exp = adj - (ndig - 1)
    quantum = Decimal(1).scaleb(quant_exp)
    up = dv.quantize(quantum, rounding=ROUND_CEILING)
    return str(up)

B_decimal_25 = decimal_ceiling(B_up_mpf, 25)

# ---- (5) sanity -------------------------------------------------------------
B_f = float(B_upper)
primal_lp = None
primal_feas = None
if args.report:
    try:
        rep = json.load(open(args.report))
        primal_lp = float(rep.get("score_lp"))
        primal_feas = float(rep.get("primal_feasible"))
    except Exception as e:
        print(f"[warn] could not read report {args.report}: {e}")

# Sanity gates (no cross-reach monotonicity is invoked): B must be a positive
# bound strictly under the trivial box ceiling and dominate both measured primals.
sane_positive = B_f > 0.0
sane_ceiling = B_f < args.ceiling_sanity
sane_ge_lp = (primal_lp is None) or (B_f >= primal_lp)
sane_ge_feas = (primal_feas is None) or (B_f >= primal_feas)

print("\n=== CERTIFIED BOUND ===")
print(f"B (certified upper, float)   = {B_f:.16g}")
print(f"B decimal (25 digits, up)    = {B_decimal_25}")
print(f"term1 = TOL*sum_y (nearest)  = {float(term1_upper):.16g}")
print(f"10*sum|r| (certified upper)  = {float(term2_iv.b):.6e}")
print(f"sum|r| (certified upper)     = {float(sum_r_upper):.6e}")
print("\n=== SANITY ===")
print(f"B > 0                          : {sane_positive}")
print(f"B < ceiling_sanity={args.ceiling_sanity}        : {sane_ceiling}")
if primal_lp is not None:
    print(f"B >= LP primal({K})={primal_lp:.16g}  : {sane_ge_lp}  gap={B_f-primal_lp:.3e}")
if primal_feas is not None:
    print(f"B >= feas primal({K})={primal_feas:.16g}: {sane_ge_feas}  gap={B_f-primal_feas:.3e}")

all_sane = sane_positive and sane_ceiling and sane_ge_lp and sane_ge_feas

# ---- hash of the dual weights ----------------------------------------------
h = hashlib.sha256()
h.update(m_arr.tobytes()); h.update(Y_arr.tobytes())
h.update(str(denom_pow).encode()); h.update(str(K).encode())
y_hash = h.hexdigest()

cert = {
    "problem": "EinsteinArena PNT board (id 7) — CEILING theorem, weak-duality certificate",
    "scope_K": K,
    "scope_M": M_dom,                     # constraint-domain upper: m in [1, 10K-1]
    "TOL": TOL,
    "box": [-10, 10],
    "bound_statement": f"S*({K}) <= {B_decimal_25}",
    "B_upper_decimal_25_roundup": B_decimal_25,
    # B_nearest_float / term1_..._nearest_float print the NEAREST float of a
    # two-sided quantity, NOT a one-sided certified bound; only the fields named
    # "..._certified_upper..." (and the exact-rational vault) are one-sided.
    "B_nearest_float": repr(B_f),
    "term1_TOL_times_sum_y_nearest_float": repr(float(term1_upper)),
    "sum_abs_r_certified_upper_float": repr(float(sum_r_upper)),
    "ten_times_sum_abs_r_certified_upper_float": repr(float(term2_iv.b)),
    "sum_y_exact_fraction": [S_Y, D],
    "sum_y_float": repr(float(sum_y_exact)),
    "denom_pow": denom_pow,
    "n_dual_rows": len(m_arr),
    "n_negative_Y_clipped": neg,
    "n_columns_k": K - 1,
    "y_hash_sha256": y_hash,
    "y_source_npz": args.npz,
    "mp_prec_bits": args.prec,
    "diagnostics_float": {
        "sum_abs_r_float": sum_abs_r_float,
        "max_abs_r_float": max_abs_r_float,
    },
    "sanity": {
        "ceiling_sanity": args.ceiling_sanity,
        "B_lt_ceiling": bool(sane_ceiling),
        "B_positive": bool(sane_positive),
        "primal_LP": primal_lp,
        "primal_feasible": primal_feas,
        "B_ge_primal_LP": bool(sane_ge_lp),
        "B_ge_primal_feasible": bool(sane_ge_feas),
        "all_sane": bool(all_sane),
    },
    "scope_ledger": {
        "certifies": f"S*({K}) for the FULL integer-constraint program (all m in [1,{M_dom}], "
                     f"TOL=1.0001, keys 2..{K}, boxes [-10,10], verifier f(1) adjustment).",
        "monotone_note": MONOTONE_NOTE,
    },
}
with open(args.out, "w") as f:
    json.dump(cert, f, indent=1)
print(f"\nwrote {args.out}")
print(f"y_hash={y_hash}")
print(f"ALL_SANE={all_sane}")
sys.exit(0 if all_sane else 3)
