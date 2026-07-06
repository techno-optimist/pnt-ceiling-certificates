#!/usr/bin/env python
"""Lambda-space projection refiner for the S*(K) dual certificate.

y_m (m in support, sorted) is reparametrized as y_i = lam_i - lam_{i+1}
(lam_{ns+1}=0, lam = suffix sums of y). Then
   sum_m y_m a_{m,k} = sum_i lam_i * D_{i,k},
   D_{i,k} = a_{m_i,k} - a_{m_{i-1},k}
           = (#multiples of k in (m_{i-1}, m_i]) - (m_i - m_{i-1})/k
(divisor-counting rows: well-conditioned, unlike consecutive a-rows).
Project lam0 onto {D^T lam = c} via Cholesky of D^T D with iterative refinement,
map back to y, clip noise-scale negatives, re-project. B(y) valid for ANY y>=0.
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

class DOp:
    """D (ns x nk): D_i = a(m_i) - a(m_{i-1}), m_0 = 0 (a(0)=0). Chunked or stored."""
    def __init__(self, ms, ks, chunk=1024, store_max_gb=8.0):
        self.ms = ms; self.ks = ks; self.chunk = chunk
        self.prev = np.concatenate([[0], ms[:-1]]).astype(np.int64)
        need = ms.size * ks.size * 8 / 1e9
        self.D = None
        if need <= store_max_gb:
            t0 = time.time()
            self.D = np.empty((ms.size, ks.size))
            for i in range(0, ms.size, chunk):
                self.D[i:i + chunk] = self._blk(i, min(i + chunk, ms.size))
            log(f"stored dense D ({need:.2f} GB) in {time.time()-t0:.1f}s")
        else:
            log(f"D needs {need:.1f} GB > cap; implicit chunked ops")

    def _blk(self, i0, i1):
        mi = self.ms[i0:i1, None]; mp = self.prev[i0:i1, None]
        kk = self.ks[None, :]; kf = self.ks.astype(np.float64)[None, :]
        return ((mi // kk) - (mp // kk)).astype(np.float64) - (mi - mp).astype(np.float64) / kf

    def Dt_lam(self, lam):
        if self.D is not None:
            return lam @ self.D
        out = np.zeros(self.ks.size)
        for i in range(0, self.ms.size, self.chunk):
            out += lam[i:i + self.chunk] @ self._blk(i, min(i + self.chunk, self.ms.size))
        return out

    def D_u(self, u):
        if self.D is not None:
            return self.D @ u
        out = np.empty(self.ms.size)
        for i in range(0, self.ms.size, self.chunk):
            out[i:i + self.chunk] = self._blk(i, min(i + self.chunk, self.ms.size)) @ u
        return out

    def gram(self):
        if self.D is not None:
            return self.D.T @ self.D
        nk = self.ks.size
        Gm = np.zeros((nk, nk))
        for i in range(0, self.ms.size, self.chunk):
            X = self._blk(i, min(i + self.chunk, self.ms.size))
            Gm += X.T @ X
        return Gm

def a_resid(ms, y, ks, c, chunk=1024):
    """r = c - sum_m y_m a_{m,k} straight from the original formula."""
    racc = np.zeros(ks.size)
    kk = ks[None, :]; kf = ks.astype(np.float64)[None, :]
    for i in range(0, ms.size, chunk):
        mi = ms[i:i + chunk, None]
        aa = (mi // kk).astype(np.float64) - mi.astype(np.float64) / kf
        racc += y[i:i + chunk] @ aa
    return c - racc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yfile", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bind-tol", type=float, default=1e-6)
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--inner-refine", type=int, default=2)
    ap.add_argument("--store-max-gb", type=float, default=18.0)
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
        log(f"binding rows: {binding.size}; y-support: {ms0.size}")
        supp |= set(binding.tolist())
    ms = np.array(sorted(supp), dtype=np.int64)
    ns = ms.size
    log(f"support union ns={ns}, nk={nk}")

    y = np.zeros(ns)
    pos = {int(m): i for i, m in enumerate(ms)}
    for m, v in zip(ms0, ys0):
        y[pos[int(m)]] = v
    r0 = a_resid(ms, y, ks, c, args.chunk)
    B0 = TOL * y.sum() + 10 * np.abs(r0).sum()
    log(f"start: B={B0:.10f} sum|r|={np.abs(r0).sum():.3e} max|r|={np.abs(r0).max():.3e}")

    op = DOp(ms, ks, chunk=args.chunk, store_max_gb=args.store_max_gb)
    t0 = time.time()
    Gm = op.gram()
    log(f"gram built ({Gm.nbytes/1e9:.2f} GB) in {time.time()-t0:.1f}s")
    diag = np.diag(Gm)
    log(f"gram diag range: [{diag.min():.3e}, {diag.max():.3e}]")
    ridge = 0.0
    dmean = float(diag.mean())
    for attempt in range(5):
        try:
            t0 = time.time()
            A = Gm if ridge == 0 else Gm + ridge * np.eye(nk)
            CF = cho_factor(A, lower=True, check_finite=False)
            log(f"cholesky ok (ridge={ridge:.2e}) in {time.time()-t0:.1f}s")
            break
        except np.linalg.LinAlgError:
            ridge = dmean * 1e-14 if ridge == 0 else ridge * 100
            log(f"cholesky failed; ridge -> {ridge:.2e}")
    else:
        raise SystemExit("cholesky failed")

    def solve_refined(rho):
        """(D^T D) u = rho with iterative refinement (residual in float64)."""
        u = cho_solve(CF, rho, check_finite=False)
        for _ in range(args.inner_refine):
            res = rho - op.Dt_lam(op.D_u(u))
            u = u + cho_solve(CF, res, check_finite=False)
        return u

    lam = np.cumsum(y[::-1])[::-1].copy()   # suffix sums
    for cyc in range(args.cycles):
        rho = c - op.Dt_lam(lam)
        u = solve_refined(rho)
        dl = op.D_u(u)
        lam = lam + dl
        yv = lam - np.append(lam[1:], 0.0)
        miny = float(yv.min()); nneg = int((yv < 0).sum())
        yv = np.maximum(yv, 0.0)
        # re-canonicalize lam from clipped y
        lam = np.cumsum(yv[::-1])[::-1].copy()
        r = a_resid(ms, yv, ks, c, args.chunk)
        B = TOL * yv.sum() + 10 * np.abs(r).sum()
        log(f"cycle {cyc}: |dlam|max={np.abs(dl).max():.3e} miny={miny:.3e} "
            f"nneg={nneg} sum|r|={np.abs(r).sum():.3e} max|r|={np.abs(r).max():.3e} "
            f"B={B:.10f}")
    y = yv

    # dyadic round 2^48, final residual straight from the a-formula
    Y = np.rint(y * (2.0 ** 48)).astype(np.int64)
    keep = Y > 0
    msd = ms[keep]; Yd = Y[keep]
    yd = Yd.astype(np.float64) / (2.0 ** 48)
    r = a_resid(msd, yd, ks, c, args.chunk)
    sumy = float(yd.sum())
    sabs = float(np.abs(r).sum()); mabs = float(np.abs(r).max())
    B = TOL * sumy + 10 * sabs
    log(f"final: B={B:.10f} sumy={sumy:.10f} support={int(keep.sum())} "
        f"sum|r|={sabs:.6e} max|r|={mabs:.3e}")
    log(f"improvement: {B0:.10f} -> {B:.10f}")

    kw = dict(m=msd, Y=Yd, denom_pow=48, y=yd, B_float=B,
              sum_abs_r=sabs, max_abs_r=mabs, K=K, M=M, TOL=TOL)
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
