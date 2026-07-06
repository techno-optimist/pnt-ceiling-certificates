#!/usr/bin/env python
"""Projection refiner for the S*(K) dual certificate.

Given the float LP dual y0 >= 0 (support S = {y0>0} U {binding rows of primal f}),
project y0 onto the affine set {y : A_S^T y = c} (nearest point, so signs are
preserved up to noise scale), clip tiny negatives, re-project; a few cycles drive
r_k = c_k - sum_m y_m a_{m,k} to ~1e-12 across ALL k while keeping y >= 0.

B(y) = 1.0001*sum(y) + 10*sum|r| is valid for ANY y >= 0 (weak duality).

Memory-lean: never stores G at production scale unless it fits; Gram = G^T G
(nk x nk) accumulated in m-chunks; Cholesky (scipy) with tiny ridge fallback.
"""
import numpy as np, argparse, os, time, json
from scipy.linalg import cho_factor, cho_solve

def log(m): print(f"[{time.strftime('%H:%M:%S')}]", m, flush=True)

def full_scan(f, ks, M):
    dd = np.zeros(M + 1)
    for j in range(ks.size):
        fk = f[j]
        if fk != 0.0:
            k = int(ks[j]); dd[k::k] += fk
    S1 = float((f / ks).sum())
    return np.cumsum(dd)[1:] - np.arange(1, M + 1, dtype=np.float64) * S1

def a_block(ms_blk, ks):
    """rows m in ms_blk x cols k: a_{m,k} = floor(m/k) - m/k, float64."""
    mi = ms_blk[:, None]
    return (mi // ks[None, :]).astype(np.float64) - mi.astype(np.float64) / ks.astype(np.float64)[None, :]

class Op:
    """Implicit A_S (ns x nk) with chunked products."""
    def __init__(self, ms, ks, chunk=1024, store=None):
        self.ms, self.ks, self.chunk = ms, ks, chunk
        self.G = store  # optional dense (ns x nk)

    def At_y(self, y):  # returns (nk,) = sum_m y_m a_{m,:}
        if self.G is not None:
            return y @ self.G
        out = np.zeros(self.ks.size)
        for i in range(0, self.ms.size, self.chunk):
            out += y[i:i + self.chunk] @ a_block(self.ms[i:i + self.chunk], self.ks)
        return out

    def A_u(self, u):   # returns (ns,) = a_{m,:} . u
        if self.G is not None:
            return self.G @ u
        out = np.empty(self.ms.size)
        for i in range(0, self.ms.size, self.chunk):
            out[i:i + self.chunk] = a_block(self.ms[i:i + self.chunk], self.ks) @ u
        return out

    def gram(self):     # (nk x nk) = sum_m outer(a_m, a_m)
        nk = self.ks.size
        if self.G is not None:
            return self.G.T @ self.G
        Gm = np.zeros((nk, nk))
        for i in range(0, self.ms.size, self.chunk):
            X = a_block(self.ms[i:i + self.chunk], self.ks)
            Gm += X.T @ X
        return Gm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yfile", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bind-tol", type=float, default=1e-6)
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--store-G-max-gb", type=float, default=8.0)
    ap.add_argument("--chunk", type=int, default=1024)
    args = ap.parse_args()

    K = args.K; M = 10 * K; TOL = 1.0001
    d = np.load(args.yfile)
    ms0 = d["m"].astype(np.int64); ys0 = d["y"].astype(np.float64)
    f = d["f"].astype(np.float64) if "f" in d.files else None
    ks = np.arange(2, K + 1, dtype=np.int64)
    nk = ks.size
    c = -np.log(ks.astype(np.float64)) / ks

    supp = set(ms0.tolist())
    if f is not None:
        E = full_scan(f, ks, M)
        binding = np.nonzero(E > TOL - args.bind_tol)[0] + 1
        log(f"binding rows from primal: {binding.size}; y-support: {ms0.size}")
        supp |= set(binding.tolist())
    ms = np.array(sorted(supp), dtype=np.int64)
    ns = ms.size
    log(f"support union: ns={ns}, nk={nk}")

    store = None
    need_gb = ns * nk * 8 / 1e9
    if need_gb <= args.store_G_max_gb:
        t0 = time.time()
        store = np.empty((ns, nk))
        for i in range(0, ns, args.chunk):
            store[i:i + args.chunk] = a_block(ms[i:i + args.chunk], ks)
        log(f"stored dense G ({need_gb:.2f} GB) in {time.time()-t0:.1f}s")
    else:
        log(f"G would need {need_gb:.1f} GB > cap; using implicit chunked ops")
    op = Op(ms, ks, chunk=args.chunk, store=store)

    y = np.zeros(ns)
    pos = {int(m): i for i, m in enumerate(ms)}
    for m, v in zip(ms0, ys0):
        y[pos[int(m)]] = v
    r = c - op.At_y(y)
    B0 = TOL * y.sum() + 10 * np.abs(r).sum()
    log(f"start: B={B0:.10f} sum|r|={np.abs(r).sum():.3e} max|r|={np.abs(r).max():.3e}")

    t0 = time.time()
    Gm = op.gram()
    log(f"gram built ({Gm.nbytes/1e9:.2f} GB) in {time.time()-t0:.1f}s")
    dmean = float(np.mean(np.diag(Gm)))
    ridge = 0.0
    for attempt in range(4):
        try:
            t0 = time.time()
            CF = cho_factor(Gm if ridge == 0 else Gm + ridge * np.eye(nk),
                            lower=True, overwrite_a=(attempt > 0), check_finite=False)
            log(f"cholesky ok (ridge={ridge:.2e}) in {time.time()-t0:.1f}s")
            break
        except np.linalg.LinAlgError:
            ridge = dmean * (1e-14 if ridge == 0 else ridge / dmean * 100)
            log(f"cholesky failed; retrying with ridge={ridge:.2e}")
    else:
        raise SystemExit("cholesky failed at all ridges")

    for cyc in range(args.cycles):
        rho = c - op.At_y(y)
        u = cho_solve(CF, rho, check_finite=False)
        delta = op.A_u(u)
        y = y + delta
        miny = float(y.min())
        nneg = int((y < 0).sum())
        y = np.maximum(y, 0.0)
        r = c - op.At_y(y)
        B = TOL * y.sum() + 10 * np.abs(r).sum()
        log(f"cycle {cyc}: |delta|_max={np.abs(delta).max():.3e} miny={miny:.3e} "
            f"nneg_clipped={nneg} sum|r|={np.abs(r).sum():.3e} "
            f"max|r|={np.abs(r).max():.3e} B={B:.10f}")

    # dyadic round 2^48 and final evaluation
    Y = np.rint(y * (2.0 ** 48)).astype(np.int64)
    keep = Y > 0
    msd = ms[keep]; Yd = Y[keep]
    yd = np.zeros(ns); yd[keep] = Yd.astype(np.float64) / (2.0 ** 48)
    r = c - op.At_y(yd)
    sumy = float(yd.sum())
    sabs = float(np.abs(r).sum()); mabs = float(np.abs(r).max())
    B = TOL * sumy + 10 * sabs
    log(f"final: B={B:.10f} sumy={sumy:.10f} support={int(keep.sum())} "
        f"sum|r|={sabs:.6e} max|r|={mabs:.3e}")
    log(f"improvement: {B0:.10f} -> {B:.10f}")

    kw = dict(m=msd, Y=Yd, denom_pow=48, y=Yd.astype(np.float64) / (2.0 ** 48),
              B_float=B, sum_abs_r=sabs, max_abs_r=mabs, K=K, M=M, TOL=TOL)
    if f is not None:
        kw["f"] = f
    if "working_rows" in d.files:
        kw["working_rows"] = d["working_rows"]
    np.savez_compressed(args.out, **kw)
    rep = dict(B_float=B, sumy=sumy, support=int(keep.sum()), sum_abs_r=sabs,
               max_abs_r=mabs, B_start=B0, out=args.out)
    with open(args.out + ".json", "w") as fh:
        json.dump(rep, fh, indent=1)
    log("DONE " + json.dumps(rep))

if __name__ == "__main__":
    main()
