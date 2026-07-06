#!/usr/bin/env python
"""MEASUREMENT (NON-CERTIFIED): platform-verifier end-to-end run.

This is an illustrative MEASUREMENT script, NOT part of the certified proof and
NOT on the `make verify` path. It requires the external EinsteinArena client
module `arena.arena_client` (the competition platform's client), which is NOT
shipped in this repository; the script therefore does not run from the archive
alone. It builds the alpha-rescaled 2000-key submission vector from a y_final
npz and runs the platform's REAL verifier on it, reporting the exact score and
pass/fail versus the 1.0001 cap and the gate. Nothing here enters any certified
bound; the certified path is scripts/recompute_bound.py + scripts/log_audit.py
(numpy + mpmath only). See the note's measurement ledger.

usage: verify_sub.py <y_final.npz> [out.json] [extra_safety=1.0]
"""
import numpy as np, json, sys
sys.path.insert(0, ".")  # measurement-stage: needs the EinsteinArena client module
from arena.arena_client import ArenaClient

GATE = 0.9973984360068646
CAP = 1.0001

yfile = sys.argv[1]
outjson = sys.argv[2] if len(sys.argv) > 2 else None
extra = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

d = np.load(yfile)
f = d["f"].astype(np.float64)
K = int(d["K"]); M = int(d["M"]); TOL = float(d["TOL"])
ks = None
# reconstruct the whitelist keys: the driver stored f aligned to ks=sorted whitelist.
# recover keys from working set is not stored; but f length == len(whitelist).
# We must know the key ordering. The driver used ks = sorted(set(whitelist))&[2,K].
# It is NOT saved in y_final, so pass keys via sidecar if present.
import os
kf = os.path.join(os.path.dirname(yfile), "keys.json")
if os.path.exists(kf):
    wl = np.array(sorted({int(k) for k in json.load(open(kf))["keys"]}), dtype=np.int64)
    wl = wl[(wl >= 2) & (wl <= K)]
    assert wl.size == f.size, f"keys {wl.size} != f {f.size}"
    ks = wl
else:
    # fall back: full dense 2..K (only valid for the uncapped solve)
    ks = np.arange(2, K + 1, dtype=np.int64)
    assert ks.size == f.size, f"dense keys {ks.size} != f {f.size}; need keys.json sidecar"

cmin = np.log(ks.astype(np.float64)) / ks


def full_scan(fv, kv, Mtop):
    dd = np.zeros(Mtop + 1)
    for j in range(kv.size):
        fk = fv[j]
        if fk != 0.0:
            k = int(kv[j]); dd[k::k] += fk
    S1 = float((fv / kv).sum())
    return np.cumsum(dd)[1:] - np.arange(1, Mtop + 1, dtype=np.float64) * S1


# reach for the verifier = 10 * max key
maxkey = int(ks.max())
reach = 10 * maxkey
E = full_scan(f, ks, reach)
maxE = float(E.max())
score_raw = float(-(f * cmin).sum())
# alpha to bring maxE strictly under CAP with buffer
margin = 2e-6 * extra
alpha = (CAP - margin) / maxE if maxE > CAP - margin else 1.0
fsub = alpha * f
Esub = full_scan(fsub, ks, reach)
maxEsub = float(Esub.max())
score_sub = float(-(fsub * cmin).sum())
print(f"raw: maxkey={maxkey} reach={reach} maxE={maxE:.10f} score={score_raw:.10f}")
print(f"alpha={alpha:.12f} -> maxEsub={maxEsub:.10f} score_sub={score_sub:.10f}")

sub = {str(int(k)): float(v) for k, v in zip(ks, fsub) if abs(v) > 1e-15}
print(f"nonzero keys in submission: {len(sub)}  (all |f|<=10: {all(abs(v)<=10 for v in sub.values())})")

# REAL verifier
p = ArenaClient.__dict__  # noqa
c = ArenaClient(agent_name="CHRONOS")
prob = c.get_problem("prime-number-theorem")
res = ArenaClient.run_verifier(prob["verifier"], {"partial_function": sub})
print("=== REAL VERIFIER RESULT ===")
print(repr(res)[:800])
exact = None
if isinstance(res, (int, float)):
    exact = float(res)
elif isinstance(res, dict):
    exact = res.get("score", res.get("value"))
    try:
        exact = float(exact)
    except Exception:
        pass
print(f"exact score = {exact}")
print(f"gate = {GATE}   clears_gate = {exact is not None and isinstance(exact,float) and exact > GATE}")

if outjson:
    json.dump({"partial_function": sub}, open(outjson, "w"))
    print(f"wrote {outjson} ({len(sub)} keys)")
