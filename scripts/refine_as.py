#!/usr/bin/env python
"""Active-set lambda-space refiner for the S*(K) dual certificate.

Support points ms (sorted). y >= 0 on support; lam_i = sum_{j>=i} y_j.
   sum_m y_m a_{m,k} = sum_i lam_i D_{i,k},
   D_{i,k} = (floor(m_i/k)-floor(m_{i-1}/k)) - (m_i-m_{i-1})/k   (m_0=0)
Projection of lam onto {D^T lam = c} via Cholesky of D^T D + inner refinement.
If some y_i = lam_i - lam_{i+1} goes negative, we DROP that support point
(merge intervals) and re-project -- never clip real mass (Sigma|r| charges
~0.4*nk per unit of clipped mass). Gram is updated incrementally on drops.
Result: y >= 0 exactly, residuals at solve precision. B(y) valid for ANY y>=0.
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

def rows_D(mp, mi, ks):
    """D rows for intervals (mp, mi]: shape (len, nk)."""
    mp = mp[:, None]; mi = mi[:, None]
    kk = ks[None, :]; kf = ks.astype(np.float64)[None, :]
    return ((mi // kk) - (mp // kk)).astype(np.float64) - (mi - mp).astype(np.float64) / kf

class DOp:
    def __init__(self, ms, ks, chunk=1024):
        self.ks = ks; self.chunk = chunk
        self.set_support(ms)

    def set_support(self, ms):
        self.ms = ms
        self.prev = np.concatenate([[0], ms[:-1]]).astype(np.int64)

    def Dt_lam(self, lam):
        out = np.zeros(self.ks.size)
        for i in range(0, self.ms.size, self.chunk):
            j = min(i + self.chunk, self.ms.size)
            out += lam[i:j] @ rows_D(self.prev[i:j], self.ms[i:j], self.ks)
        return out

    def D_u(self, u):
        out = np.empty(self.ms.size)
        for i in range(0, self.ms.size, self.chunk):
            j = min(i + self.chunk, self.ms.size)
            out[i:j] = rows_D(self.prev[i:j], self.ms[i:j], self.ks) @ u
        return out

    def gram(self):
        nk = self.ks.size
        Gm = np.zeros((nk, nk))
        for i in range(0, self.ms.size, self.chunk):
            j = min(i + self.chunk, self.ms.size)
            X = rows_D(self.prev[i:j], self.ms[i:j], self.ks)
            Gm += X.T @ X
        return Gm

def a_resid(ms, y, ks, c, chunk=1024):
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
    ap.add_argument("--max-passes", type=int, default=30)
    ap.add_argument("--inner-refine", type=int, default=2)
    ap.add_argument("--drop-floor", type=float, default=1e-13)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--check-gram", action="store_true")
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
    log(f"support union ns={ms.size}, nk={nk}")

    y = np.zeros(ms.size)
    pos = {int(m): i for i, m in enumerate(ms)}
    for m, v in zip(ms0, ys0):
        y[pos[int(m)]] = v
    r0 = a_resid(ms, y, ks, c, args.chunk)
    B0 = TOL * y.sum() + 10 * np.abs(r0).sum()
    log(f"start: B={B0:.10f} sum|r|={np.abs(r0).sum():.3e} max|r|={np.abs(r0).max():.3e}")

    op = DOp(ms, ks, chunk=args.chunk)
    t0 = time.time()
    Gm = op.gram()
    log(f"gram built ({Gm.nbytes/1e9:.2f} GB) in {time.time()-t0:.1f}s")

    best = None
    for pss in range(args.max_passes):
        dmean = float(np.mean(np.diag(Gm)))
        ridge = 0.0
        while True:
            try:
                t0 = time.time()
                A = Gm.copy() if ridge == 0 else Gm + ridge * np.eye(nk)
                CF = cho_factor(A, lower=True, overwrite_a=True, check_finite=False)
                break
            except np.linalg.LinAlgError:
                ridge = dmean * 1e-14 if ridge == 0 else ridge * 100
                if ridge > dmean * 1e-4:
                    raise SystemExit("cholesky hopeless")
        tchol = time.time() - t0

        lam = np.cumsum(y[::-1])[::-1].copy()
        rho = c - op.Dt_lam(lam)
        pre_inf = float(np.abs(rho).max())
        u = cho_solve(CF, rho, check_finite=False)
        for _ in range(args.inner_refine):
            res = rho - op.Dt_lam(op.D_u(u))
            u = u + cho_solve(CF, res, check_finite=False)
        lam = lam + op.D_u(u)
        rho2 = c - op.Dt_lam(lam)
        post_inf = float(np.abs(rho2).max())
        yv = lam - np.append(lam[1:], 0.0)
        drop = yv < -args.drop_floor
        nneg = int(drop.sum())
        minny = float(yv.min())
        # honest evaluation of the clipped candidate at this pass
        ycl = np.maximum(yv, 0.0)
        r = a_resid(ms, ycl, ks, c, args.chunk)
        B = TOL * ycl.sum() + 10 * np.abs(r).sum()
        log(f"pass {pss}: ns={ms.size} chol={tchol:.1f}s ridge={ridge:.1e} "
            f"|rho|inf pre={pre_inf:.3e} post={post_inf:.3e} miny={minny:.3e} "
            f"drops={nneg} B={B:.10f} sum|r|={np.abs(r).sum():.3e}")
        if best is None or B < best["B"]:
            best = dict(B=B, ms=ms.copy(), y=ycl.copy())
        if nneg == 0:
            log("no drops needed -- converged")
            break
        # ---- drop support points, merge intervals, update gram ----
        keep = ~drop
        ms_new = ms[keep]
        prev_old = op.prev
        # positions whose row changes: all dropped + survivors following a dropped run
        surv_after = []
        keep_idx = np.nonzero(keep)[0]
        drop_idx = np.nonzero(drop)[0]
        dropped_set = set(drop_idx.tolist())
        for i in keep_idx:
            if i - 1 >= 0 and (i - 1) in dropped_set:
                surv_after.append(i)
        surv_after = np.array(surv_after, dtype=np.int64)
        # old rows to remove: dropped rows + old versions of surv_after rows
        rm_pos = np.concatenate([drop_idx, surv_after]).astype(np.int64)
        X_old_mp = prev_old[rm_pos]; X_old_mi = ms[rm_pos]
        # new rows: surv_after with new prev = previous surviving point (or 0)
        new_prev = []
        for i in surv_after:
            j = i - 1
            while j >= 0 and j in dropped_set:
                j -= 1
            new_prev.append(ms[j] if j >= 0 else 0)
        new_prev = np.array(new_prev, dtype=np.int64)
        t0 = time.time()
        for i0 in range(0, rm_pos.size, args.chunk):
            j0 = min(i0 + args.chunk, rm_pos.size)
            X = rows_D(X_old_mp[i0:j0], X_old_mi[i0:j0], ks)
            Gm -= X.T @ X
        for i0 in range(0, surv_after.size, args.chunk):
            j0 = min(i0 + args.chunk, surv_after.size)
            X = rows_D(new_prev[i0:j0], ms[surv_after[i0:j0]], ks)
            Gm += X.T @ X
        log(f"  gram updated ({rm_pos.size} old rows out, {surv_after.size} new rows in) "
            f"in {time.time()-t0:.1f}s")
        y = np.maximum(yv[keep], 0.0)
        ms = ms_new
        op.set_support(ms)
        if args.check_gram:
            Gt = op.gram()
            err = float(np.abs(Gt - Gm).max())
            log(f"  gram check: max err {err:.3e}")
            Gm = Gt

    ms_b = best["ms"]; y_b = best["y"]
    Y = np.rint(y_b * (2.0 ** 48)).astype(np.int64)
    kp = Y > 0
    msd = ms_b[kp]; Yd = Y[kp]
    yd = Yd.astype(np.float64) / (2.0 ** 48)
    r = a_resid(msd, yd, ks, c, args.chunk)
    sumy = float(yd.sum())
    sabs = float(np.abs(r).sum()); mabs = float(np.abs(r).max())
    B = TOL * sumy + 10 * sabs
    log(f"final: B={B:.10f} sumy={sumy:.10f} support={int(kp.sum())} "
        f"sum|r|={sabs:.6e} max|r|={mabs:.3e}")
    log(f"improvement: {B0:.10f} -> {B:.10f}")

    kw = dict(m=msd, Y=Yd, denom_pow=48, y=yd, B_float=B,
              sum_abs_r=sabs, max_abs_r=mabs, K=K, M=M, TOL=TOL)
    if f is not None:
        kw["f"] = f
    if "working_rows" in d.files:
        kw["working_rows"] = d["working_rows"]
    np.savez_compressed(args.out, **kw)
    rep = dict(B_float=B, sumy=sumy, support=int(kp.sum()), sum_abs_r=sabs,
               max_abs_r=mabs, B_start=B0, out=args.out)
    with open(args.out + ".json", "w") as fh:
        json.dump(rep, fh, indent=1)
    log("DONE " + json.dumps(rep))

if __name__ == "__main__":
    main()
