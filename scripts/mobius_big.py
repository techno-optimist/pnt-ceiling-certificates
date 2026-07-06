import numpy as np, time
def mobius(N):
    mu=np.ones(N+1,dtype=np.int8); is_c=np.zeros(N+1,bool); primes=[]; mu[0]=0
    for i in range(2,N+1):
        if not is_c[i]: primes.append(i); mu[i]=-1
        for p in primes:
            if i*p>N: break
            is_c[i*p]=True
            if i%p==0: mu[i*p]=0; break
            else: mu[i*p]=-mu[i]
    mu[1]=1; return mu

# ---- (A) identity sum_{k<=x} mu(k) floor(x/k) == 1 ----
Xmax=1_000_000; mu=mobius(Xmax)
muL=mu.astype(np.int64)
bad=0
for x in [1,2,3,7,10,100,999,1000,12345,100000,500000,999983,1_000_000]:
    s=int(np.sum(muL[1:x+1]*(x//np.arange(1,x+1))))
    if s!=1: bad+=1; print("  IDENTITY FAIL x=",x,"->",s)
print(f"(A) identity sum_k mu(k)floor(x/k)=1 checked at 13 x up to 1e6: violations={bad}")

def analyze(K):
    mu=mobius(10*K)  # need mu up to K only for construction; 10K for nothing extra, but reuse
    muK=mu[:K+1].astype(np.int64)
    ks=np.arange(1,K+1)
    T=float(np.sum(muK[1:]/ks))                      # sum_{k=1}^K mu(k)/k
    S1=float(np.sum(muK[2:]/np.arange(2,K+1)))       # sum_{k>=2} mu(k)/k = T-1
    # score of raw truncated mobius = -sum_{k=2}^K mu(k) ln k/k
    lk=np.log(np.arange(2,K+1))
    score=float(-np.sum(muK[2:]*lk/np.arange(2,K+1)))
    # max E over m in [1,10K]  (feasibility of raw truncated Mobius, uncapped)
    M=10*K; dd=np.zeros(M+1)
    t0=time.time()
    for k in range(2,K+1):
        v=muK[k]
        if v!=0: dd[k::k]+=v
    cs=np.cumsum(dd)
    mm=np.arange(0,M+1)
    E=cs-mm*S1
    Emax=float(E[1:].max()); at=int(np.argmax(E[1:]))+1
    Emin=float(E[1:].min())
    nnz=int(np.count_nonzero(muK[2:]))
    print(f"\nK={K}: nnz keys(2..K)={nnz}, T=sum mu/k={T:.8e}, S1(k>=2)={S1:.8e}")
    print(f"   raw-truncated-Mobius score = {score:.10f}   (1.0001*mu -> {1.0001*score:.10f})")
    print(f"   max_m E(m) over [1,{M}] = {Emax:.6f} at m={at}; min E={Emin:.6f}")
    print(f"   feasible(<=1.0001)? {Emax<=1.0001}   [{ '#keys>2000 anyway: ' + str(nnz>2000) }]  (build {time.time()-t0:.1f}s)")
    return score,Emax,nnz

for K in (48000,64000):
    analyze(K)
