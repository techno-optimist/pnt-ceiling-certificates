#!/bin/bash
# HARVEST K: refine y_final_K{K}.npz down toward ~1e-11 residuals, exact-certify
# the min-B candidate, independently skeptic-verify. Non-fatal on individual
# refiner failure. Usage: harvest_ladder.sh <K>
K="${1:-12000}"
cd "$(dirname "$0")" || exit 1
PY="${PY:-python3}"
YF="y_final_K${K}.npz"
REP="report_K${K}.json"
echo "=== HARVEST K${K} start $(date -u) ==="
if [ ! -f "$YF" ]; then echo "MISSING $YF -- abort"; touch "HARVEST_FAILED_K${K}"; exit 1; fi

# baseline exact cert of the driver's built-in-1e-9-polished output
echo "--- baseline certify $YF ---"
$PY certify_ladder.py "$YF" "cert_K${K}_base.json" --report "$REP" || true

# refiners (sequential; failures/timeouts non-fatal). lam & as use divisor rows
# (best-conditioned, produced the banked 2.34e-11 at K=4800); proj uses a-rows.
for R in lam proj as; do
  echo "--- refine_${R} K${K} $(date -u) ---"
  timeout 5400 $PY "refine_${R}.py" --yfile "$YF" --K "$K" --out "yref_${R}_K${K}.npz" \
    || echo "refine_${R} FAILED/timeout"
done

# pick the min exact-proxy-B candidate among base + refined
$PY - "$K" <<'PYEOF'
import sys, glob, numpy as np
K=int(sys.argv[1])
cands=[f"y_final_K{K}.npz"]+sorted(glob.glob(f"yref_*_K{K}.npz"))
best=None
for c in cands:
    try:
        d=np.load(c); B=float(d["B_float"])
        mr=float(d["max_abs_r"]) if "max_abs_r" in d.files else float("nan")
        print(f"cand {c}: B_float={B:.12f} max|r|={mr:.3e} support={d['m'].size}")
        if best is None or B<best[1]: best=(c,B)
    except Exception as e:
        print(f"cand {c}: ERR {e}")
print("WINNER", best[0], best[1])
open(f"winner_K{K}.txt","w").write(best[0])
PYEOF
WIN=$(cat "winner_K${K}.txt")

echo "=== EXACT certify winner $WIN ==="
$PY certify_ladder.py "$WIN" "certificate_K${K}.json" --report "$REP"
CERTRC=$?
echo "=== SKEPTIC verify winner $WIN ==="
$PY skeptic_ladder.py "$WIN" "certificate_K${K}.json" --report "$REP"
SKRC=$?
echo "certify_rc=$CERTRC skeptic_rc=$SKRC (0=all-sane/all-pass; certify:3=sanity-fail; skeptic:4=verdict-fail)"
echo "=== HARVEST K${K} done $(date -u) ==="
touch "HARVEST_DONE_K${K}"
