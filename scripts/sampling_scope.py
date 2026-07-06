import numpy as np, mpmath as mp
mp.mp.dps=40
NS=10_000_000
def scope(maxkey):
    ub=10*maxkey
    W=mp.mpf(ub)-1                 # measure of support [1, ub]
    Nint=ub-1                      # unit intervals [m,m+1), m=1..ub-1  (== #full-program constraints monitorable)
    p0=(1-1/W)**NS                 # P(a given interval gets 0 of NS samples)
    exp_missed=Nint*p0
    lam=mp.mpf(NS)/W               # mean samples per interval
    print(f"maxkey(reach)={maxkey}: upper_bound={ub}, #intervals={Nint}, mean samples/interval lam={float(lam):.4f}")
    print(f"   P(interval unmonitored)=(1-1/{ub-1})^1e7 = {mp.nstr(p0,4)}")
    print(f"   EXPECTED unmonitored constraints = {mp.nstr(exp_missed,5)}   (Poisson approx N*e^-lam = {float(Nint*mp.e**-lam):.5g})")
    return ub,Nint

for mk in (48000,64000):
    scope(mk)

# ---- exact realized count for the ACTUAL RandomState(42) stream at reach=48000 ----
print("\nRealized (seed 42) unmonitored-interval count:")
for mk in (48000,64000):
    ub=10*mk
    rng=np.random.RandomState(42)
    x=rng.uniform(1,ub,size=NS)     # same stream as verifier (batch concat == single draw)
    fl=np.floor(x).astype(np.int64) # sample lands in interval [fl, fl+1)
    seen=np.zeros(ub+2,bool); seen[fl]=True
    # monitorable integer constraints m in [1, ub-1]
    m=np.arange(1,ub)               # ub-1 of them
    missed=int(np.count_nonzero(~seen[m]))
    print(f"   reach={mk}: realized #unmonitored intervals (seed42) = {missed}  of {ub-1}")
