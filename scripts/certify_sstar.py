#!/usr/bin/env python3
"""
CERTIFY phase: exact certified weak-duality upper bound on S*(K) for the
EinsteinArena PNT board (problem id 7).

Input: a dual weight file (npz) with fields
   m           : int64[nrows]   support row indices (integers in [1, 10K])
   Y           : int64[nrows]   dyadic numerators, y_m = Y_m / 2^denom_pow
   denom_pow   : int            dyadic denominator exponent (D = 2^denom_pow)
   K, M, TOL   : problem params (M = 10*K, TOL = 1.0001)

Bound (weak duality, VALID for ANY y >= 0):
   B(y) = TOL * sum_m y_m  +  10 * sum_{k=2}^{K} |r_k|,   B(y) >= S*(K)
where
   r_k = c_k - sum_m y_m a_{m,k},   c_k = -ln(k)/k,   a_{m,k} = floor(m/k) - m/k.

CERTIFIED ARITHMETIC (no float on the bound path):
 * y_m = Y_m / D exact (D a power of two).
 * a_{m,k} = (k*floor(m/k) - m)/k exact rational; k*floor(m/k)-m = -(m mod k)
   is an exact integer in (-(k-1), 0].
 * q_k = sum_m y_m a_{m,k} = N_k / (k*D)  with N_k = sum_m Y_m*(k*floor(m/k)-m)
   an exact integer (accumulated in Python big-ints; per-term product fits int64
   and is proven non-overflowing, see OVERFLOW note).
 * r_k = -( ln(k) + N_k/D ) / k, so |r_k| = |ln(k) + N_k/D| / k.
   ln(k) enclosed by an outward-rounded mpmath interval (mp.iv.log). N_k/D and
   1/k are exact. The whole bound is assembled in mp.iv interval arithmetic and
   we output the UPPER endpoint -> a certified upper bound on S*(K).

OVERFLOW note (int64 safety of the per-term product):
   |k*floor(m/k) - m| = (m mod k) <= k-1 <= K-1 < 5000.
   max Y_m ~ 1.4e14. product <= 1.4e14 * 5000 = 7e17 < 9.22e18 = int64 max. OK.
   The accumulation N_k (up to ~3e21) is done in Python int (arbitrary precision).
"""
import sys, json, hashlib, time
import numpy as np
from fractions import Fraction
from mpmath import mp, iv

NPZ = sys.argv[1] if len(sys.argv) > 1 else "y_polish1e-09_K4800.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "certificate.json"

# ---- interval precision: 300 bits >> 72 (N_k) + 48 (D) + margin -------------
mp.prec = 400
iv.prec = 400

d = np.load(NPZ)
m_arr = d["m"].astype(np.int64)
Y_arr = d["Y"].astype(np.int64)
denom_pow = int(d["denom_pow"])
K = int(d["K"]); M = int(d["M"]); TOL = float(d["TOL"])
D = 1 << denom_pow  # 2^denom_pow, exact

assert M == 10 * K, f"M ({M}) must equal 10*K ({10*K})"
assert TOL == 1.0001

# ---- (1) y -> exact dyadic; clip negatives to 0 (preserves cert validity) ---
neg = int((Y_arr < 0).sum())
if neg:
    print(f"[clip] {neg} negative Y clipped to 0")
    Y_arr = np.where(Y_arr < 0, 0, Y_arr)
assert (Y_arr >= 0).all()
# rows must lie in [1, M]
assert m_arr.min() >= 1 and m_arr.max() <= M, (int(m_arr.min()), int(m_arr.max()), M)
# max product safety check (empirical, matches the analytic bound)
maxYval = int(Y_arr.max())
assert maxYval * (K - 1) < (1 << 63), "int64 overflow risk in per-term product"

S_Y = int(Y_arr.sum())                 # exact integer  = sum_m Y_m
sum_y_exact = Fraction(S_Y, D)         # exact rational = sum_m y_m
print(f"K={K} M={M} rows={len(m_arr)} D=2^{denom_pow} sum_y={float(sum_y_exact):.15g}")

# ---- (2) exact integer residual pass over ALL k = 2..K ----------------------
# Vectorized-but-exact: per-term product Y_m*(k*floor(m/k)-m) fits int64
# (proven above), and the final reduction is done in Python big-ints via an
# object-dtype sum so the accumulation (up to ~3e21) never overflows.
Y_obj = Y_arr.astype(object)   # 4800 Python ints, computed once

t0 = time.time()
iv_D = iv.mpf(D)
sum_r_iv = iv.mpf(0)            # running interval for sum_k |r_k|
sum_abs_r_float = 0.0          # float mirror (diagnostics only, NOT on cert path)
max_abs_r_float = 0.0

report_interval = 500
for k in range(2, K + 1):
    # delta_m = k*floor(m/k) - m = -(m mod k)  (int64, |.|<=k-1<K, exact)
    delta = k * (m_arr // k) - m_arr                 # int64 vector, exact
    # N_k exact big int: object-dtype product+sum (no int64 accumulation)
    Nk = int((Y_obj * delta).sum())
    # |r_k| = |ln(k) + N_k/D| / k
    lnk = iv.log(k)                          # outward-rounded enclosure of ln(k)
    Ak = iv.mpf(Nk) / iv_D                   # exact-ish (outward) N_k/D
    val = lnk + Ak                           # interval enclosing ln(k)+N_k/D
    # |val| upper bound = max(|a|,|b|); build [0, upper] then /k
    ub = iv.mpf(max(abs(val.a), abs(val.b)))
    abs_r = ub / iv.mpf(k)                   # interval upper bound on |r_k|
    sum_r_iv = sum_r_iv + abs_r
    # diagnostics (float, off-path)
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
# term1 = TOL * sum_y   (TOL = 10001/10000 exact; sum_y = S_Y/D exact)
term1_iv = iv.mpf(10001 * S_Y) / iv.mpf(10000 * D)
term2_iv = iv.mpf(10) * sum_r_iv
B_iv = term1_iv + term2_iv
B_upper = B_iv.b                          # certified upper bound (interval sup)

# certified sum|r| upper and TOL*sum_y for reporting
sum_r_upper = sum_r_iv.b
term1_upper = term1_iv.b

# 25-digit directed-UP decimal of B_upper
mp.prec = 400
B_up_mpf = mp.mpf(B_upper)
# round UP to 25 significant digits
from mpmath import nstr
B_str_25 = mp.nstr(B_up_mpf, 25, strip_zeros=False)
# a guaranteed-upper decimal at 25 digits: add one ulp at the 25th place if needed
# (nstr rounds to nearest; bump last digit to be safe -> still an upper bound)
def decimal_ceiling(x_mpf, ndig):
    # produce a decimal string >= x with ndig significant digits, rounding UP
    import mpmath as _m
    s = _m.nstr(x_mpf, ndig+3, strip_zeros=False)
    # parse and round up at ndig sig digits using Decimal
    from decimal import Decimal, ROUND_CEILING, getcontext
    getcontext().prec = ndig + 5
    dv = Decimal(s)
    # quantize to ndig significant figures via exponent
    q = dv.scaleb(0)
    # find exponent of leading digit
    adj = dv.adjusted()  # exponent of most significant digit
    quant_exp = adj - (ndig - 1)
    quantum = Decimal(1).scaleb(quant_exp)
    up = dv.quantize(quantum, rounding=ROUND_CEILING)
    return str(up)

B_decimal_25 = decimal_ceiling(B_up_mpf, 25)

# ---- (5) sanity: compare to primals -----------------------------------------
PRIMAL_K4800_LP   = 0.9963683211217781   # from report_K4800.json (this program)
PRIMAL_K4800_FEAS = 0.9963682137701979   # feasibility-scanned alpha primal
LEADER_K48000     = 0.9973457049300725   # JSAgent, board leader (max key 48000)
CHRONOS_K48000    = 0.99651773           # our own verifier-accepted (max key 48000)

B_f = float(B_upper)
print("\n=== CERTIFIED BOUND ===")
print(f"B (certified upper, float)   = {B_f:.16g}")
print(f"B decimal (25 digits, up)    = {B_decimal_25}")
print(f"term1 = TOL*sum_y (upper)    = {float(term1_upper):.16g}")
print(f"10*sum|r| (upper)            = {float(iv.mpf(10)*sum_r_iv).b if False else float(term2_iv.b):.6e}")
print(f"sum|r| (certified upper)     = {float(sum_r_upper):.6e}")
print(f"\nSanity vs K=4800 program primal:")
print(f"  B >= LP primal(4800)  {B_f} >= {PRIMAL_K4800_LP}: {B_f >= PRIMAL_K4800_LP}  gap={B_f-PRIMAL_K4800_LP:.3e}")
print(f"  B >= feas primal(4800) {B_f} >= {PRIMAL_K4800_FEAS}: {B_f >= PRIMAL_K4800_FEAS}  gap={B_f-PRIMAL_K4800_FEAS:.3e}")
print(f"\nScope check vs K=48000 board (this bound does NOT cover K=48000):")
print(f"  B(4800) vs leader(48000) {B_f} vs {LEADER_K48000}: B<leader = {B_f < LEADER_K48000} (EXPECTED, S* increasing in K)")

# ---- hash of the dual weights ----------------------------------------------
h = hashlib.sha256()
h.update(m_arr.tobytes()); h.update(Y_arr.tobytes())
h.update(str(denom_pow).encode()); h.update(str(K).encode())
y_hash = h.hexdigest()

cert = {
    "problem": "EinsteinArena PNT board (id 7) — CEILING theorem, weak-duality certificate",
    "scope_K": K,
    "scope_M": M,
    "TOL": TOL,
    "box": [-10, 10],
    "bound_statement": f"S*({K}) <= {B_decimal_25}",
    "B_upper_decimal_25_roundup": B_decimal_25,
    "B_upper_float": repr(B_f),
    "term1_TOL_times_sum_y_upper_float": repr(float(term1_upper)),
    "sum_abs_r_certified_upper_float": repr(float(sum_r_upper)),
    "ten_times_sum_abs_r_upper_float": repr(float(term2_iv.b)),
    "sum_y_exact_fraction": [S_Y, D],
    "sum_y_float": repr(float(sum_y_exact)),
    "denom_pow": denom_pow,
    "n_dual_rows": len(m_arr),
    "n_negative_Y_clipped": neg,
    "n_columns_k": K - 1,
    "y_hash_sha256": y_hash,
    "y_source_npz": NPZ,
    "mp_prec_bits": 400,
    "diagnostics_float": {
        "sum_abs_r_float": sum_abs_r_float,
        "max_abs_r_float": max_abs_r_float,
    },
    "sanity": {
        "primal_K4800_LP": PRIMAL_K4800_LP,
        "primal_K4800_feasible": PRIMAL_K4800_FEAS,
        "B_ge_primal_LP": bool(B_f >= PRIMAL_K4800_LP),
        "B_ge_primal_feasible": bool(B_f >= PRIMAL_K4800_FEAS),
        "duality_gap_vs_LP": B_f - PRIMAL_K4800_LP,
        "duality_gap_vs_feasible": B_f - PRIMAL_K4800_FEAS,
    },
    "scope_ledger": {
        "certifies": f"S*({K}) for the FULL integer-constraint program (all m in [1,{M}], "
                     f"TOL=1.0001, keys 2..{K}, boxes [-10,10], verifier f(1) adjustment).",
        "does_NOT_certify": "S*(48000) (the live board reach). S*(K) is nondecreasing in K, "
                            "so this K=4800 ceiling does NOT upper-bound S*(48000); the "
                            "K=48000 production dual solve had not converged at certify time "
                            "(round 1 of dual simplex, no valid K=48000 dual checkpoint yet).",
        "leader_K48000": LEADER_K48000,
        "chronos_K48000": CHRONOS_K48000,
        "note": "B(4800) < leader(48000) is EXPECTED and consistent (leader uses keys up to 48000).",
    },
}
with open(OUT, "w") as f:
    json.dump(cert, f, indent=1)
print(f"\nwrote {OUT}")
print(f"y_hash={y_hash}")
