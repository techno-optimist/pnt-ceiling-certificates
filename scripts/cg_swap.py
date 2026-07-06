#!/usr/bin/env python
"""Restricted column-generation swap loop for the 2000-key PNT extraction.

Start from an initial whitelist + its converged restricted solve (y_final npz).
Each iteration:
  1. price ALL keys 2..K: r_k = cmax_k - sum_m y_m*(floor(m/k) - m/k)   (reduced cost)
  2. new whitelist = top-`keep` current keys by |f|  UNION  top-`add` excluded keys by |r_k|,
     capped at CAP (2000). Always force-keep the dense low band 2..LOWMAX.
  3. resolve via solve_restricted.py at --K K, --tol TOL.
Records score each iteration; stops after N iters or when improvement < eps.
"""
import numpy as np, json, os, subprocess, sys, time

PY = "python3"  # solve-stage helper; set to your interpreter
DRIVER = "solve_restricted.py"
BASE = "cg"
K = 12000
M = 10 * K
TOL = 1.00009999
CAP = 2000
LOWMAX = 60          # force-keep keys 2..60 (they carry the bulk of the objective)
KEEP = 1500          # keep this many current keys by |f|
ADD = 500            # add this many excluded keys by |reduced cost|
NITERS = 6
EPS = 5e-6

ks_all = np.arange(2, K + 1, dtype=np.int64)
cmin_all = np.log(ks_all.astype(np.float64)) / ks_all
cmax_all = -cmin_all


def residuals(ms, ys, ks, cmax, chunk=256):
    racc = np.zeros(ks.size)
    kk = ks[None, :]
    kf = ks.astype(np.float64)[None, :]
    for i in range(0, ms.size, chunk):
        mi = ms[i:i + chunk, None]
        aa = (mi // kk).astype(np.float64) - mi.astype(np.float64) / kf
        racc += ys[i:i + chunk] @ aa
    return cmax - racc


def price(yfile):
    """r_k for all keys 2..K from the saved dual in yfile."""
    d = np.load(yfile)
    ms = d["m"].astype(np.int64)           # support of dual (m values, 1-indexed)
    ys = d["y"].astype(np.float64)
    r = residuals(ms, ys, ks_all, cmax_all)   # r_k = reduced cost of key k
    # current primal f over the WHITELIST that produced this solve
    f = d["f"].astype(np.float64)          # aligned to that solve's ks (whitelist, sorted)
    return r, f, d


def build_new_whitelist(cur_keys, cur_f, r):
    cur_keys = np.asarray(cur_keys)
    absf = np.abs(cur_f)
    # keep top-KEEP current keys by |f|
    keep_order = cur_keys[np.argsort(-absf)][:KEEP]
    kept = set(int(k) for k in keep_order)
    # force low band
    kept |= set(range(2, LOWMAX + 1))
    # excluded keys by |r|
    cur_set = set(int(k) for k in cur_keys)
    excl_mask = np.array([int(k) not in cur_set for k in ks_all])
    excl_keys = ks_all[excl_mask]
    excl_r = np.abs(r[excl_mask])
    add_order = excl_keys[np.argsort(-excl_r)]
    for k in add_order:
        if len(kept) >= CAP:
            break
        kept.add(int(k))
    # if still under cap, top up from remaining current keys
    if len(kept) < CAP:
        for k in cur_keys[np.argsort(-absf)]:
            if len(kept) >= CAP:
                break
            kept.add(int(k))
    return sorted(kept)


def run_solve(keys, outdir):
    os.makedirs(outdir, exist_ok=True)
    kf = os.path.join(outdir, "keys.json")
    json.dump({"keys": keys}, open(kf, "w"))
    done = os.path.join(outdir, "DONE"); fail = os.path.join(outdir, "FAILED")
    for m in (done, fail):
        if os.path.exists(m):
            os.remove(m)
    cmd = [PY, DRIVER, "--K", str(K), "--keys-file", kf, "--tol", repr(TOL),
           "--round1-solver", "ipm", "--resolve-solver", "ipm", "--run-crossover", "on",
           "--round1-time-limit", "900", "--round-time-limit", "600",
           "--max-round-retries", "4", "--add-cap", "120000", "--threads", "6",
           "--polish-time-limit", "600", "--outdir", outdir,
           "--done-marker", done, "--fail-marker", fail]
    log = open(os.path.join(outdir, "run.log"), "w")
    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    rep = os.path.join(outdir, f"report_K{K}.json")
    if not os.path.exists(rep):
        return None
    return json.load(open(rep))


def main():
    init_keys_file = sys.argv[1]   # json {"keys":[...]}
    init_yfile = sys.argv[2]       # y_final npz from the initial solve
    os.makedirs(BASE, exist_ok=True)
    cur_keys = json.load(open(init_keys_file))["keys"]
    yfile = init_yfile
    best = {"score": -1, "iter": -1, "yfile": None, "keys": cur_keys}
    # score of the init solve
    d0 = np.load(init_yfile)
    s0 = float(d0["primal_feasible"])
    print(f"[cg] iter0 init score(primal_feasible)={s0:.10f} score_lp={float(d0['score_lp']):.10f} "
          f"nkeys={len(cur_keys)} maxkey={max(cur_keys)}", flush=True)
    best = {"score": s0, "iter": 0, "yfile": init_yfile, "keys": cur_keys}
    prev = s0
    for it in range(1, NITERS + 1):
        r, f, d = price(yfile)
        # current keys aligned to this solve's whitelist (sorted)
        cur_keys_arr = np.array(sorted(set(int(k) for k in cur_keys)))
        # f is aligned to the solved whitelist (ks filtered to [2,K]); rebuild
        solved_keys = cur_keys_arr[(cur_keys_arr >= 2) & (cur_keys_arr <= K)]
        if f.size != solved_keys.size:
            print(f"[cg] WARN f.size={f.size} vs solved_keys={solved_keys.size}; realigning by trim")
            n = min(f.size, solved_keys.size)
            solved_keys = solved_keys[:n]; f = f[:n]
        new_keys = build_new_whitelist(solved_keys, f, r)
        outdir = os.path.join(BASE, f"it{it}")
        t0 = time.time()
        rep = run_solve(new_keys, outdir)
        if rep is None:
            print(f"[cg] iter{it} SOLVE FAILED (no report); stop", flush=True)
            break
        sc = float(rep["primal_feasible"]); sclp = float(rep["score_lp"])
        yfile = rep["y_file"]
        cur_keys = new_keys
        maxk = max(new_keys)
        print(f"[cg] iter{it} score(primal_feasible)={sc:.10f} score_lp={sclp:.10f} "
              f"maxkey={maxk} nkeys={len(new_keys)} maxE={rep['maxE_scan']:.8f} "
              f"dt={time.time()-t0:.0f}s", flush=True)
        if sc > best["score"]:
            best = {"score": sc, "iter": it, "yfile": yfile, "keys": new_keys}
        if sc - prev < EPS and sc <= prev:
            print(f"[cg] converged/plateau at iter{it} (gain {sc-prev:.2e})", flush=True)
            prev = sc
            # continue one extra iter in case of oscillation? no, stop.
            break
        prev = sc
    json.dump({"best_iter": best["iter"], "best_score": best["score"],
               "best_yfile": best["yfile"], "best_keys_file": os.path.join(BASE, "best_keys.json")},
              open(os.path.join(BASE, "cg_summary.json"), "w"))
    json.dump({"keys": best["keys"]}, open(os.path.join(BASE, "best_keys.json"), "w"))
    print(f"[cg] BEST iter={best['iter']} score={best['score']:.10f} "
          f"yfile={best['yfile']}", flush=True)
    open(os.path.join(BASE, "CG_DONE"), "w").write("done")


if __name__ == "__main__":
    main()
