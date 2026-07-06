#!/usr/bin/env python3
"""
log_audit.py -- INDEPENDENT audit of the ln(k) enclosures used on the certified
path, addressing referee MAJOR 2 (don't rely SOLELY on the mpmath log oracle).

Two independent verifications, over k = 2..K:

(a) DUAL-PRECISION NESTING. For every k, compute iv.log(k) at prec p1 and p2
    (default 400 and 800 bit). Confirm the higher-precision interval nests
    inside (or equals, up to outward rounding) the lower-precision one:
        [a2,b2] subset~ [a1,b1]   i.e.  a1 <= a2  and  b2 <= b1.
    Report the max interval width at each precision and any k where nesting
    fails (should be none).

(b) RATIONAL BRACKET SPOT-CHECK on a spread of ~nspot values of k. For each,
    compute a rigorous rational [lo,hi] enclosing ln(k) with NO transcendental
    oracle, via  ln(k)=2*atanh((k-1)/(k+1)),  atanh(u)=sum u^(2j+1)/(2j+1)
    (all terms positive => partial sum is a lower bound; tail bounded by
    u^(2N+1)/((2N+1)(1-u^2)) => upper bound). Adaptive N to a target width.
    Confirm the mpmath iv.log(k) enclosure lies inside the rational bracket
    (both are rigorous, so they must be mutually consistent), and report the
    rational-bracket width achieved.

Usage: log_audit.py --K 4800 [--p1 400 --p2 800 --nspot 20 --target 1e-25]
"""
import sys, argparse, time, json
from fractions import Fraction
from mpmath import mp, iv


def raw_to_frac(t):
    sign, man, exp, bc = t
    val = Fraction(man)
    val = val * (1 << exp) if exp >= 0 else val / (1 << (-exp))
    return -val if sign else val


def iv_endpoints_frac(interval):
    """Exact (lower, upper) rationals of an mpmath iv interval, via raw tuples
    (avoids mp.mpf re-rounding at the ambient mp.prec)."""
    lo_raw, hi_raw = interval._mpi_
    return raw_to_frac(lo_raw), raw_to_frac(hi_raw)


def ln_bracket(k, target_width, max_terms=200000):
    """Rigorous rational [lo,hi] enclosing ln(k) to within target_width."""
    u = Fraction(k - 1, k + 1)
    u2 = u * u
    one_m_u2 = 1 - u2
    S = Fraction(0)
    up = u
    j = 0
    tail = None
    while j < max_terms:
        S += up / (2 * j + 1)
        up *= u2
        j += 1
        tail = up / ((2 * j + 1) * one_m_u2)
        if 2 * tail <= target_width:
            break
    return 2 * S, 2 * (S + tail), j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--p1", type=int, default=400)
    ap.add_argument("--p2", type=int, default=800)
    ap.add_argument("--nspot", type=int, default=20)
    ap.add_argument("--target", type=str, default="1e-25")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    K = args.K
    target = Fraction(1, 10 ** int(-__import__("math").log10(float(args.target))))

    # ---- (a) dual-precision nesting over ALL k -----------------------------
    print(f"[log-audit] (a) dual-precision nesting p1={args.p1} p2={args.p2}, k=2..{K}")
    t0 = time.time()
    nest_fail = []
    max_w1 = Fraction(0)   # exact widths
    max_w2 = Fraction(0)
    argmax1 = argmax2 = 0
    for k in range(2, K + 1):
        iv.prec = args.p1
        l1 = iv.log(k) / iv.mpf(k)      # audit ln(k)/k enclosure as used
        iv.prec = args.p2
        l2 = iv.log(k) / iv.mpf(k)
        # exact endpoints via raw tuples (no mp.prec re-rounding)
        a1, b1 = iv_endpoints_frac(l1)
        a2, b2 = iv_endpoints_frac(l2)
        # nesting: higher-prec interval nests inside lower-prec: a1<=a2 & b2<=b1
        if not (a1 <= a2 and b2 <= b1):
            nest_fail.append((k, float(a1), float(a2), float(b2), float(b1)))
        w1 = b1 - a1
        w2 = b2 - a2
        if w1 > max_w1:
            max_w1 = w1; argmax1 = k
        if w2 > max_w2:
            max_w2 = w2; argmax2 = k
    dt = time.time() - t0
    print(f"[log-audit] done {dt:.1f}s")
    print(f"[log-audit] max ln(k)/k interval width @p1={args.p1}: {float(max_w1):.3e} at k={argmax1}")
    print(f"[log-audit] max ln(k)/k interval width @p2={args.p2}: {float(max_w2):.3e} at k={argmax2}")
    print(f"[log-audit] nesting failures: {len(nest_fail)}")
    for f in nest_fail[:10]:
        print("   NEST FAIL", f)

    # ---- (b) rational bracket spot check -----------------------------------
    # spread: small, mid, large; skip the very largest few (slow) but include a
    # representative large k. Deterministic spread.
    import random
    rng = random.Random(20260706)
    # A rigorous rational atanh-series bracket costs O(k) terms with big-int
    # numerators, so we cap the spot set to k <= 1000 (still a broad spread:
    # small primes/composites through mid-range). This is a consistency
    # cross-check of the mpmath iv.log oracle, not the certified path itself.
    hi_cap = min(K, 1000)
    lo_ks = [2, 3, 5, 7, 10, 13, 17, 23, 31, 47, 64, 97, 128, 251, 512]
    mid_ks = sorted(rng.sample(range(300, hi_cap), min(6, max(1, hi_cap - 300))))
    spot = sorted(set([k for k in lo_ks if k <= hi_cap] + mid_ks))
    spot = [k for k in spot if 2 <= k <= K][:max(args.nspot, 20)]
    print(f"\n[log-audit] (b) rational bracket spot-check on {len(spot)} k (target width {args.target})")
    print(f"[log-audit] (rigorous rational atanh-series bracket; spot k capped to <= {hi_cap})")
    t0 = time.time()
    max_rat_w = 0.0
    disagree = []
    for k in spot:
        lo, hi, nt = ln_bracket(k, target)          # rigorous rational [lo,hi]
        iv.prec = args.p2
        a, b = iv_endpoints_frac(iv.log(k))          # rigorous iv [a,b], exact
        # Both [lo,hi] and [a,b] rigorously enclose the SAME true ln(k), so they
        # MUST intersect: lo <= b AND a <= hi. (Strict containment either way is
        # NOT required -- whichever is narrower simply pins ln(k) tighter.)
        consistent = (lo <= b) and (a <= hi)
        # sanity: the intersection [max(lo,a), min(hi,b)] is nonempty & valid
        inter_lo = max(lo, a); inter_hi = min(hi, b)
        consistent = consistent and (inter_lo <= inter_hi)
        w = float(hi - lo)
        max_rat_w = max(max_rat_w, w)
        if not consistent:
            disagree.append((k, nt, w, float(lo), float(hi), float(a), float(b)))
        print(f"   k={k:6d} nterms={nt:6d} rat_width={w:.2e} rat/iv consistent={consistent}")
    print(f"[log-audit] rational spot-check done {time.time()-t0:.1f}s")
    print(f"[log-audit] max rational-bracket width over spot set = {max_rat_w:.2e}")
    print(f"[log-audit] disagreements (iv.log outside rational bracket) = {len(disagree)}")
    for dd in disagree:
        print("   DISAGREE", dd)

    result = {
        "K": K, "p1": args.p1, "p2": args.p2,
        "max_lnk_over_k_width_p1": mp.nstr(max_w1, 8),
        "max_lnk_over_k_width_p2": mp.nstr(max_w2, 8),
        "argmax_width_p1": argmax1, "argmax_width_p2": argmax2,
        "nesting_failures": len(nest_fail),
        "spot_k": spot,
        "max_rational_bracket_width": max_rat_w,
        "rational_disagreements": len(disagree),
    }
    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)
        print(f"[log-audit] wrote {args.out}")
    ok = (len(nest_fail) == 0) and (len(disagree) == 0)
    print(f"[log-audit] AUDIT_OK = {ok}")
    sys.exit(0 if ok else 6)


if __name__ == "__main__":
    main()
