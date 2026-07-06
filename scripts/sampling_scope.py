#!/usr/bin/env python
"""MEASUREMENT (Proposition "realized coverage of the deployed sampler").

Deployed-sampler coverage computation. This is a *measurement* script, not on
the certified `make verify` path; it reproduces the deterministic coverage
figures quoted in Proposition~\\ref{prop:coverage} of the note.

The deployed EinsteinArena verifier accepts a submission with largest key
K_s, then draws NUM_SAMPLES uniform reals x ~ Uniform[1, 10*K_s) with a FIXED
pseudorandom seed and rejects the submission if the raw floor constraint is
violated at any drawn x. Because floor(x) is constant on each unit interval
[m, m+1), a sample at x tests exactly the integer constraint at m = floor(x)
(Lemma "the sampled verifier evaluates integer constraints"). So the set of
integer constraints the deployed sampler actually monitors is
{ floor(x) : x in the sample stream }, and its "coverage" of the constraint
domain m in [1, 10*K_s - 1] is a deterministic property of the fixed seed.

Auditable RNG / sampler semantics (must match the deployed verifier bit-for-bit):
  * SEED         = 42                         -> numpy.random.RandomState(42)
  * NUM_SAMPLES  = 10**7                      (a single sequential draw; the
                   verifier's internal batching does not change the stream)
  * upper_bound  = 10 * maxkey                (x ~ Uniform[1, 10*maxkey))
  * sample -> interval index  m = floor(x)    (int64)
  * constraint domain monitored: m in [1, 10*maxkey - 1]  (== 10K-1)

A reader can confirm the sampler below is bit-for-bit the deployed stream by
checking VERIFIER_SAMPLER_SHA256 (printed at run time) against the frozen
source string VERIFIER_SAMPLER_SRC; the realized-coverage loop calls exactly
that frozen source, so the hash provably pins the code that produced the
numbers.
"""
import hashlib
import numpy as np
import mpmath as mp

mp.mp.dps = 40

# ---- fixed, auditable sampler parameters (mirror the deployed verifier) ------
SEED = 42
NUM_SAMPLES = 10_000_000                       # 10**7
NS = NUM_SAMPLES                               # legacy alias used below

# Corollary 7 uses reaches 4800 and 12000; 48000 and 64000 are the current /
# near-term board reaches. All four are reported for BOTH the expected-missed
# computation and the realized (seed-42) coverage check.
REACHES = (4800, 12000, 48000, 64000)

# ---- FROZEN verifier-sampler source (hash-pinned) ----------------------------
# This string IS the sampler used by the realized-coverage loop below (it is
# exec'd, then called), so VERIFIER_SAMPLER_SHA256 pins the exact code that runs.
VERIFIER_SAMPLER_SRC = (
    "def deployed_sample_floors(maxkey, seed, num_samples):\n"
    "    # x ~ Uniform[1, 10*maxkey); interval index m = floor(x), int64.\n"
    "    upper_bound = 10 * maxkey\n"
    "    rng = np.random.RandomState(seed)\n"
    "    x = rng.uniform(1, upper_bound, size=num_samples)\n"
    "    return np.floor(x).astype(np.int64), upper_bound\n"
)
VERIFIER_SAMPLER_SHA256 = hashlib.sha256(VERIFIER_SAMPLER_SRC.encode()).hexdigest()
exec(VERIFIER_SAMPLER_SRC)  # defines deployed_sample_floors(...) from the frozen source

# sanity: the parameters we document are the ones the frozen sampler consumes.
assert SEED == 42 and NUM_SAMPLES == 10**7


def expected_missed(maxkey):
    """Expected number of unmonitored unit intervals under NS iid uniforms."""
    ub = 10 * maxkey
    W = mp.mpf(ub) - 1                          # measure of support [1, ub)
    Nint = ub - 1                               # unit intervals [m,m+1), m=1..ub-1
    p0 = (1 - 1 / W) ** NS                      # P(a given interval gets 0 samples)
    exp_missed = Nint * p0
    lam = mp.mpf(NS) / W                        # mean samples per interval
    print(f"reach(maxkey)={maxkey}: upper_bound={ub}, #intervals(=10K-1)={Nint}, "
          f"mean samples/interval lam={float(lam):.4f}")
    print(f"   P(interval unmonitored)=(1-1/{ub-1})^1e7 = {mp.nstr(p0,4)}")
    print(f"   EXPECTED unmonitored constraints = {mp.nstr(exp_missed,5)}   "
          f"(Poisson approx N*e^-lam = {float(Nint*mp.e**-lam):.5g})")
    return ub, Nint


print(f"[sampler] frozen-source SHA-256 = {VERIFIER_SAMPLER_SHA256}")
print(f"[sampler] SEED={SEED}  NUM_SAMPLES={NUM_SAMPLES}  x~Uniform[1, 10*maxkey)")
print("\nExpected unmonitored-interval count (all four reaches):")
for mk in REACHES:
    expected_missed(mk)

# ---- exact realized count for the ACTUAL RandomState(42) stream --------------
# Uses the hash-pinned frozen sampler above, so the numbers provably come from
# VERIFIER_SAMPLER_SHA256.
print("\nRealized (seed 42) unmonitored-interval count (all four reaches):")
for mk in REACHES:
    fl, ub = deployed_sample_floors(mk, SEED, NUM_SAMPLES)  # frozen sampler
    seen = np.zeros(ub + 2, bool)
    seen[fl] = True
    m = np.arange(1, ub)                        # constraint domain m in [1, 10K-1]
    missed = int(np.count_nonzero(~seen[m]))
    print(f"   reach={mk}: realized #unmonitored intervals (seed42) = {missed}  of {ub-1}")
