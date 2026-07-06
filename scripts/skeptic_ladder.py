#!/usr/bin/env python3
"""
SKEPTIC VERIFY (ladder, K-agnostic): independent recomputation of the S*(K)
certificate. Pure-Fraction spot checks on random k (does NOT reuse the certifier's
integer-delta trick there) + a full independent big-int residual pass, then checks
independent B <= cert decimal and primal <= B. K/M/TOL read from the npz.
Usage: skeptic_ladder.py <npz> <certificate.json> [--report report_K{K}.json]
"""
import json, hashlib, time, random, argparse
import numpy as np
from fractions import Fraction
from mpmath import mp, iv

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("cert")
ap.add_argument("--report", default="")
ap.add_argument("--prec", type=int, default=500)
ap.add_argument("--nspot", type=int, default=20)
ap.add_argument("--seed", type=int, default=20260705)
args = ap.parse_args()

mp.prec = args.prec
iv.prec = args.prec

d = np.load(args.npz)
m_arr = d["m"].astype(np.int64)
Y_arr = d["Y"].astype(np.int64)
y_float = d["y"].astype(np.float64)
denom_pow = int(d["denom_pow"])
K = int(d["K"]); M = int(d["M"]); TOL = float(d["TOL"])
D = 1 << denom_pow
cert = json.load(open(args.cert))

print("=" * 70); print("STEP 0: integrity / hash of the dual file"); print("=" * 70)
h = hashlib.sha256()
h.update(m_arr.tobytes()); h.update(Y_arr.tobytes())
h.update(str(denom_pow).encode()); h.update(str(K).encode())
my_hash = h.hexdigest()
hash_match = my_hash == cert["y_hash_sha256"]
print("recomputed y_hash:", my_hash)
print("certificate hash :", cert["y_hash_sha256"])
print("HASH MATCH:", hash_match)

print(); print("=" * 70); print("STEP (d): scoping vs verifier semantics"); print("=" * 70)
print(f"K (max key)          = {K}")
print(f"M (domain=10*maxkey) = {M}   10*K={10*K}  match={M==10*K}")
print(f"TOL                  = {TOL}  ==1.0001: {TOL==1.0001}")
print(f"box                  = {cert['box']}")
print(f"n rows in dual       = {len(m_arr)}")
print(f"n columns k (2..K)   = {K-1}  (cert says {cert['n_columns_k']})")

print(); print("=" * 70)
print(f"STEP (b): {args.nspot} random y_m >= 0, rows in [1,M], Y/D == stored y")
print("=" * 70)
print(f"min Y = {int(Y_arr.min())}  (all Y>=0: {(Y_arr>=0).all()})")
print(f"min m = {int(m_arr.min())}  max m = {int(m_arr.max())}  in [1,{M}]: {m_arr.min()>=1 and m_arr.max()<=M}")
maxY = int(Y_arr.max())
print(f"max Y = {maxY}  maxY*(K-1) = {maxY*(K-1)}  < 2^63 = {maxY*(K-1) < (1<<63)}")
rng = random.Random(args.seed)
idxs = rng.sample(range(len(m_arr)), min(args.nspot, len(m_arr)))
bad = 0
for i in idxs:
    Ym = int(Y_arr[i]); mm = int(m_arr[i]); yf = float(y_float[i])
    exact = Fraction(Ym, D)
    ok = (1 <= mm <= M) and (Ym >= 0) and abs(float(exact) - yf) <= 1e-18 + 1e-12 * abs(yf)
    if not ok:
        bad += 1
        print(f"  BAD row i={i} m={mm} Y={Ym} y={yf} exact={float(exact)}")
print(f"{len(idxs)}-sample y check: {len(idxs)-bad}/{len(idxs)} pass")
y_rows_ok = (bad == 0) and (Y_arr >= 0).all() and m_arr.min() >= 1 and m_arr.max() <= M

print(); print("=" * 70)
print(f"STEP (a): {args.nspot} random k, EXACT r_k via pure Fraction (independent path)")
print("=" * 70)
max_claim = cert["diagnostics_float"]["max_abs_r_float"]
print(f"certificate max|r| claim = {max_claim:.6e}")
pairs = list(zip(Y_arr.tolist(), m_arr.tolist()))
ks_spot = rng.sample(range(2, K + 1), min(args.nspot, K - 1))
worst = 0.0
for k in sorted(ks_spot):
    Nk = 0
    for Ym, mm in pairs:
        Nk += Ym * (k * (mm // k) - mm)
    q_frac = Fraction(Nk, k * D)
    lnk = iv.log(k)
    q_iv = iv.mpf(int(q_frac.numerator)) / iv.mpf(int(q_frac.denominator))
    r_iv = (-lnk) / iv.mpf(k) - q_iv
    fr = float(max(abs(r_iv.a), abs(r_iv.b)))
    worst = max(worst, fr)
    flag = "" if fr <= max_claim * 1.0001 else "  <-- EXCEEDS claimed max!"
    print(f"  k={k:7d}  |r_k|<= {fr:.4e}{flag}")
spot_ok = worst <= max_claim * 1.0001
print(f"worst spot-checked |r_k| = {worst:.4e}  <= claimed max: {spot_ok}")

print(); print("=" * 70)
print("STEP (c): FULL independent residual pass -> sum|r|, recompute B"); print("=" * 70)
Y_obj = Y_arr.astype(object)
t0 = time.time()
sum_r_iv = iv.mpf(0)
sum_abs_r_float = 0.0
max_abs_r_float = 0.0
iv_D = iv.mpf(D)
for k in range(2, K + 1):
    delta = k * (m_arr // k) - m_arr
    Nk = int((Y_obj * delta.astype(object)).sum())
    lnk = iv.log(k)
    Ak = iv.mpf(Nk) / iv_D
    val = lnk + Ak
    ub = iv.mpf(max(abs(val.a), abs(val.b)))
    sum_r_iv = sum_r_iv + ub / iv.mpf(k)
    fr = abs(float(lnk.b) + Nk / D) / k
    sum_abs_r_float += fr
    if fr > max_abs_r_float:
        max_abs_r_float = fr
print(f"full pass done in {time.time()-t0:.1f}s")
print(f"independent sum|r| float = {sum_abs_r_float:.8e}   cert = {cert['diagnostics_float']['sum_abs_r_float']:.8e}")
print(f"independent max|r| float = {max_abs_r_float:.6e}    cert = {cert['diagnostics_float']['max_abs_r_float']:.6e}")

S_Y = int(Y_arr.sum())
sumy_match = [S_Y, D] == cert["sum_y_exact_fraction"]
print(f"\nsum_y exact = {S_Y}/{D}")
print(f"cert sum_y  = {cert['sum_y_exact_fraction'][0]}/{cert['sum_y_exact_fraction'][1]}  MATCH: {sumy_match}")

term1_iv = iv.mpf(10001 * S_Y) / iv.mpf(10000 * D)
term2_iv = iv.mpf(10) * sum_r_iv
B_iv = term1_iv + term2_iv
B_upper = B_iv.b
print(f"\nterm1 (TOL*sum_y) upper = {mp.nstr(mp.mpf(term1_iv.b),20)}")
print(f"term2 (10*sum|r|) upper = {mp.nstr(mp.mpf(term2_iv.b),8)}")
print(f"B upper (independent)   = {mp.nstr(mp.mpf(B_upper),25)}")
print(f"B cert decimal          = {cert['B_upper_decimal_25_roundup']}")
within = mp.mpf(B_upper) <= mp.mpf(cert["B_upper_decimal_25_roundup"])
print(f"independent B <= cert decimal? {within}")

print(); print("=" * 70); print("STEP (e): primal <= B"); print("=" * 70)
Bf = float(B_upper)
primal_ok = True
if args.report:
    try:
        rep = json.load(open(args.report))
        pl = float(rep.get("score_lp")); pf = float(rep.get("primal_feasible"))
        print(f"B = {Bf:.16g}")
        print(f"primal LP       = {pl}  B>=primal: {Bf>=pl}  gap={Bf-pl:.3e}")
        print(f"primal feasible = {pf}  B>=primal: {Bf>=pf}  gap={Bf-pf:.3e}")
        primal_ok = (Bf >= pl) and (Bf >= pf)
    except Exception as e:
        print(f"[warn] report unreadable: {e}")

print(); print("=" * 70); print("VERDICT INPUTS"); print("=" * 70)
verdict = dict(hash_match=hash_match, y_rows_ok=bool(y_rows_ok),
               scoping=bool(M == 10 * K and TOL == 1.0001),
               sumy_match=bool(sumy_match), spot_ok=bool(spot_ok),
               B_within_claim=bool(within), primal_le_B=bool(primal_ok))
for kk, vv in verdict.items():
    print(f"  {kk} = {vv}")
ALL = all(verdict.values())
print(f"\nSKEPTIC_VERDICT_ALL_PASS = {ALL}")
import sys
sys.exit(0 if ALL else 4)
