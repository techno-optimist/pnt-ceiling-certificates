#!/usr/bin/env python
"""LADDER SOLVE: S*(K) ceiling LP, hardened for K=12000 / K=24000.

Derived from solve_sstar.py (the K=4800/48000 production driver). Changes:
  * --solver {ipm,simplex}  and  --run-crossover {on,off}  (IPM+crossover gives
    clean basic duals and does NOT stall in a primal phase the way dual simplex
    did on K=48000).
  * PER-ROUND wall-clock limit (--round-time-limit). A round that hits the limit
    is a CHECKPOINT, not a FATAL: we save the HiGHS basis + working-set state,
    warm-retry the same round up to --max-round-retries times (each retry gets a
    fresh time budget), and if still not Optimal, persist a RESUMABLE marker and
    exit 0 so an external relaunch (--resume-state) can continue from the basis.
  * State saved at the TOP of every round (working set W + round index + basis),
    so resume redoes the in-flight round idempotently.

Primal / dual math identical to solve_sstar.py; certificate B(y) valid for any
y>=0 (weak duality) so every round checkpoint is a legal ceiling.
"""
import numpy as np, highspy, json, time, argparse, os, sys, traceback

INF = highspy.kHighsInf


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}]", msg, flush=True)


def residuals(ms, ys, ks, cmax, chunk=256):
    racc = np.zeros(ks.size)
    kk = ks[None, :]
    kf = ks.astype(np.float64)[None, :]
    for i in range(0, ms.size, chunk):
        mi = ms[i:i + chunk, None]
        aa = (mi // kk).astype(np.float64) - mi.astype(np.float64) / kf
        racc += ys[i:i + chunk] @ aa
    return cmax - racc


def full_scan(f, ks, M):
    dd = np.zeros(M + 1)
    for j in range(ks.size):
        fk = f[j]
        if fk != 0.0:
            k = int(ks[j])
            dd[k::k] += fk
    S1 = float((f / ks).sum())
    return np.cumsum(dd)[1:] - np.arange(1, M + 1, dtype=np.float64) * S1


def build_lp(K, M, TOL, t_upper, ks, cmin):
    nk = ks.size
    ncol = nk + M + 1
    nrow = M + 1
    mult = (M // ks).astype(np.int64)
    per_col = np.concatenate([
        mult + 1,
        np.full(M - 1, 2, dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([M + 1], dtype=np.int64),
    ])
    starts = np.zeros(ncol + 1, dtype=np.int64)
    np.cumsum(per_col, out=starts[1:])
    nnz = int(starts[-1])
    log(f"matrix: ncol={ncol} nrow={nrow} nnz={nnz}")
    index = np.empty(nnz, dtype=np.int32)
    value = np.empty(nnz, dtype=np.float64)
    for j in range(nk):
        k = int(ks[j]); p0 = int(starts[j]); cnt = int(mult[j])
        index[p0:p0 + cnt] = np.arange(k - 1, M, k, dtype=np.int32)
        value[p0:p0 + cnt] = -1.0
        index[p0 + cnt] = M
        value[p0 + cnt] = -1.0 / k
    tb = int(starts[nk])
    m1 = np.arange(1, M, dtype=np.int64)
    blk = 2 * (M - 1)
    idx2 = np.empty(blk, dtype=np.int32); val2 = np.empty(blk)
    idx2[0::2] = (m1 - 1); val2[0::2] = 1.0
    idx2[1::2] = m1;       val2[1::2] = -1.0
    index[tb:tb + blk] = idx2; value[tb:tb + blk] = val2
    pM = int(starts[nk + M - 1])
    index[pM] = M - 1; value[pM] = 1.0
    ps = int(starts[nk + M])
    index[ps:ps + M + 1] = np.arange(0, M + 1, dtype=np.int32)
    value[ps:ps + M + 1] = 1.0

    lp = highspy.HighsLp()
    lp.num_col_ = ncol
    lp.num_row_ = nrow
    lp.sense_ = highspy.ObjSense.kMinimize
    lp.col_cost_ = np.concatenate([cmin, np.zeros(M + 1)])
    lp.col_lower_ = np.concatenate([np.full(nk, -10.0), np.full(M + 1, -INF)])
    lp.col_upper_ = np.concatenate([np.full(nk, 10.0), t_upper, np.array([INF])])
    lp.row_lower_ = np.zeros(nrow)
    lp.row_upper_ = np.zeros(nrow)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = starts.astype(np.int32)
    lp.a_matrix_.index_ = index
    lp.a_matrix_.value_ = value
    return lp


def extract_certificate(h, nk, M, ks, cmax, TOL, want_residuals=True):
    sol = h.getSolution()
    if not sol.dual_valid:
        return None
    colv = np.asarray(sol.col_value)
    cold = np.asarray(sol.col_dual)
    rowd = np.asarray(sol.row_dual)
    lam = rowd[:M]
    y0 = lam - np.append(lam[1:], 0.0)
    negmass = [float(-np.minimum(c, 0.0).sum()) for c in (y0, -y0)]
    sgn = 1.0 if negmass[0] <= negmass[1] else -1.0
    y = sgn * y0
    miny = float(y.min())
    y = np.maximum(y, 0.0)
    ms_all = np.nonzero(y)[0].astype(np.int64) + 1
    ys_all = y[ms_all - 1]
    Y = np.rint(ys_all * (2.0 ** 48)).astype(np.int64)
    keep = Y > 0
    msd = ms_all[keep]; Yd = Y[keep]
    ysd = Yd.astype(np.float64) / (2.0 ** 48)
    out = dict(sign=sgn, negmass=negmass, miny_preclip=miny,
               n_support=int(msd.size),
               sumy=float(ysd.sum()),
               score_lp=float(-h.getObjectiveValue()),
               f=colv[:nk], tval=colv[nk:nk + M],
               y_m=msd, y_num=Yd, y=ysd)
    if want_residuals:
        r = residuals(msd, ysd, ks, cmax)
        rc = cold[:nk]
        xchk = min(float(np.max(np.abs(r - rc))), float(np.max(np.abs(r + rc))))
        out["r"] = r
        out["rc_crosscheck"] = xchk
        out["B_float"] = TOL * out["sumy"] + 10.0 * float(np.abs(r).sum())
        out["sum_abs_r"] = float(np.abs(r).sum())
        out["max_abs_r"] = float(np.abs(r).max())
    return out


def arm_limit(h, budget):
    """HiGHS time_limit is CUMULATIVE across run() calls on one object, so give
    the next run() `budget` more seconds by anchoring to the internal clock."""
    h.setOptionValue("time_limit", float(h.getRunTime()) + float(budget))


def configure_ipm(h, args):
    h.setOptionValue("threads", args.threads)
    h.setOptionValue("output_flag", True)
    h.setOptionValue("log_to_console", True)
    h.setOptionValue("solver", "ipm")
    h.setOptionValue("run_crossover", args.run_crossover)  # "on"/"off"/"choose"
    h.setOptionValue("time_limit", args.round1_time_limit)


def configure_simplex(h, args):
    h.setOptionValue("threads", args.threads)
    h.setOptionValue("output_flag", True)
    h.setOptionValue("log_to_console", True)
    h.setOptionValue("solver", "simplex")
    h.setOptionValue("time_limit", args.round_time_limit)


def save_state(path, W, rnd):
    """Persist working set (bit-packed) + round index for external resume."""
    np.savez_compressed(path, W_packed=np.packbits(W), W_len=W.size, rnd=rnd)


def load_state(path):
    d = np.load(path)
    W = np.unpackbits(d["W_packed"])[: int(d["W_len"])].astype(bool)
    return W, int(d["rnd"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=12000)
    p.add_argument("--outdir", default=".")
    p.add_argument("--w0-dense", type=int, default=8192)
    p.add_argument("--geo", type=float, default=1.002)
    p.add_argument("--max-rounds", type=int, default=60)
    p.add_argument("--add-cap", type=int, default=60000)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--round1-solver", choices=["ipm", "simplex"], default="ipm")
    p.add_argument("--resolve-solver", choices=["ipm", "simplex"], default="simplex")
    p.add_argument("--run-crossover", choices=["on", "off", "choose"], default="on")
    p.add_argument("--round1-time-limit", type=float, default=14400.0)
    p.add_argument("--round-time-limit", type=float, default=2700.0)
    p.add_argument("--max-round-retries", type=int, default=8)
    p.add_argument("--resume-state", default="")
    p.add_argument("--ckpt-every", type=int, default=1)
    p.add_argument("--final-dual-tol", type=float, default=1e-9)
    p.add_argument("--polish-time-limit", type=float, default=7200.0)
    p.add_argument("--done-marker", default="")   # e.g. ./DONE_K12000
    p.add_argument("--fail-marker", default="")   # e.g. ./FAILED_K12000
    args = p.parse_args()

    K = args.K; M = 10 * K; TOL = 1.0001
    tag = f"K{K}"
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    basis_path = os.path.join(outdir, f"basis_{tag}.txt")
    state_path = os.path.join(outdir, f"state_{tag}.npz")
    done_marker = args.done_marker or os.path.join(outdir, f"DONE_{tag}")
    fail_marker = args.fail_marker or os.path.join(outdir, f"FAILED_{tag}")
    resumable_marker = os.path.join(outdir, f"RESUMABLE_{tag}")

    ks = np.arange(2, K + 1, dtype=np.int64)
    nk = ks.size
    cmin = np.log(ks.astype(np.float64)) / ks
    cmax = -cmin
    log(f"START {tag} M={M} nk={nk} TOL={TOL} round1={args.round1_solver} "
        f"resolve={args.resolve_solver} crossover={args.run_crossover} "
        f"round1_tl={args.round1_time_limit}s round_tl={args.round_time_limit}s "
        f"retries={args.max_round_retries}")

    # ---- working set W (bool over m=1..M, index m) : fresh or resumed ----
    rnd0 = 1
    have_basis = False
    if args.resume_state and os.path.exists(args.resume_state):
        W, rnd0 = load_state(args.resume_state)
        log(f"RESUME from {args.resume_state}: |W|={int(W.sum())} start_round={rnd0}")
    else:
        W = np.zeros(M + 1, dtype=bool)
        dense = min(args.w0_dense, M)
        W[1:dense + 1] = True
        v = dense
        while v < M:
            v = max(v + 1, int(v * args.geo))
            if v <= M:
                W[v] = True
        W[M] = True
        log(f"W0 size = {int(W.sum())}")

    def W_to_upper(Wb):
        wm = np.nonzero(Wb[1:])[0] + 1
        tu = np.full(M, INF)
        tu[wm - 1] = TOL
        return tu

    t_build = time.time()
    lp = build_lp(K, M, TOL, W_to_upper(W), ks, cmin)
    log(f"build done in {time.time()-t_build:.1f}s")

    # Fresh start => round1_solver (IPM) for the cold round; resume (rnd0>1) is
    # always a warm simplex re-optimization from the checkpointed basis.
    resuming = bool(args.resume_state) and rnd0 > 1
    current_solver = args.resolve_solver if resuming else args.round1_solver

    h = highspy.Highs()
    if current_solver == "ipm":
        configure_ipm(h, args)
    else:
        configure_simplex(h, args)
    h.passModel(lp)
    del lp

    if resuming and os.path.exists(basis_path):
        try:
            h.readBasis(basis_path)
            have_basis = True
            log(f"read warm basis from {basis_path} (resume, solver={current_solver})")
        except Exception as e:
            log(f"readBasis failed ({e}); cold start")

    def run_round(rnd, is_ipm):
        """Solve the current LP. Simplex rounds warm-retry (basis checkpoint each
        attempt) until Optimal or retries exhausted; IPM rounds get a single shot
        (an interrupted IPM has no resumable basis, so retrying only restarts cold).
        Returns (stname, attempts)."""
        nonlocal have_basis
        retries = 1 if is_ipm else args.max_round_retries
        budget = args.round1_time_limit if is_ipm else args.round_time_limit
        for attempt in range(1, retries + 1):
            t0 = time.time()
            arm_limit(h, budget)   # fresh per-run budget (cumulative timer)
            h.run()
            st = h.getModelStatus()
            stname = h.modelStatusToString(st)
            info = h.getInfo()
            log(f"  round {rnd} attempt {attempt}: solver={'ipm' if is_ipm else 'simplex'} "
                f"status={stname} score={-h.getObjectiveValue():.10f} "
                f"iters={info.simplex_iteration_count}/ipm={info.ipm_iteration_count} "
                f"time={time.time()-t0:.1f}s |W|={int(W.sum())}")
            if stname == "Optimal":
                have_basis = True   # crossover (IPM) or simplex both leave a basis
                return stname, attempt
            if stname != "Time limit reached":
                return stname, attempt  # genuine infeasible/unbounded/error
            if is_ipm:
                return stname, attempt  # no resumable IPM state; caller hard-fails
            # simplex timeout -> checkpoint basis, warm-retry with fresh budget
            have_basis = True
            try:
                h.writeBasis(basis_path)
                log(f"  checkpoint: basis saved to {basis_path} (attempt {attempt})")
            except Exception as e:
                log(f"  writeBasis failed: {e}")
        return "Time limit reached", retries

    hist = []
    status_final = "partial"
    for rnd in range(rnd0, args.max_rounds + 1):
        is_ipm = (current_solver == "ipm")
        # checkpoint state at TOP of round (idempotent resume point)
        try:
            save_state(state_path, W, rnd)
            if have_basis:
                h.writeBasis(basis_path)
        except Exception as e:
            log(f"state/basis save failed: {e}")

        stname, attempts = run_round(rnd, is_ipm)
        if stname != "Optimal":
            if is_ipm:
                # cold IPM round did not converge in its budget. An interrupted IPM
                # solve has no basis to resume from, so this is a genuine failure
                # (raise --round1-time-limit); do NOT pretend it is resumable.
                with open(fail_marker, "w") as fh:
                    fh.write(f"round {rnd} IPM non-optimal: {stname}; "
                             f"raise --round1-time-limit")
                log(f"FATAL: round {rnd} IPM {stname} after {attempts} attempt(s); "
                    f"wrote {fail_marker}")
                sys.exit(1)
            # simplex TIMEOUT = CHECKPOINT, not fatal.
            try:
                save_state(state_path, W, rnd)
                h.writeBasis(basis_path)
            except Exception as e:
                log(f"final checkpoint save failed: {e}")
            with open(resumable_marker, "w") as fh:
                fh.write(json.dumps(dict(status=stname, round=rnd,
                                         W=int(W.sum()), K=K,
                                         basis=basis_path, state=state_path,
                                         resume_cmd=(f"--resume-state {state_path}"))))
            log(f"ROUND {rnd} simplex NOT OPTIMAL ({stname}) after {attempts} attempts. "
                f"Checkpoint written; RESUMABLE marker at {resumable_marker}. "
                f"Exiting 0 (relaunch with --resume-state {state_path}).")
            return

        sol = h.getSolution()
        tval = np.asarray(sol.col_value)[nk:nk + M]
        viol_idx = np.nonzero(tval > TOL + 1e-8)[0]
        viol_m = viol_idx + 1
        new_m = viol_m[~W[viol_m]]
        already = viol_m[W[viol_m]]
        if already.size:
            log(f"WARN: {already.size} working rows violated beyond 1e-8 "
                f"(max excess {float(tval[already-1].max()-TOL):.2e})")
        log(f"round {rnd}: violations={viol_m.size} new={new_m.size} "
            f"maxE={float(tval.max()):.8f}")
        if (rnd % args.ckpt_every == 0) or new_m.size == 0:
            cert = extract_certificate(h, nk, M, ks, cmax, TOL)
            if cert is not None:
                log(f"  ckpt: B_float={cert['B_float']:.10f} sumy={cert['sumy']:.8f} "
                    f"support={cert['n_support']} max|r|={cert['max_abs_r']:.3e} "
                    f"sum|r|={cert['sum_abs_r']:.3e} rc_xchk={cert['rc_crosscheck']:.3e} "
                    f"sign={cert['sign']} miny_preclip={cert['miny_preclip']:.3e}")
                np.savez_compressed(
                    os.path.join(outdir, f"ckpt_{tag}_r{rnd}.npz"),
                    m=cert["y_m"], Y=cert["y_num"], denom_pow=48, y=cert["y"],
                    B_float=cert["B_float"], score_lp=cert["score_lp"])
                hist.append(dict(round=rnd, W=int(W.sum()), score_lp=cert["score_lp"],
                                 B_float=cert["B_float"], sumy=cert["sumy"],
                                 support=cert["n_support"],
                                 max_abs_r=cert["max_abs_r"], sum_abs_r=cert["sum_abs_r"]))
        if new_m.size == 0:
            status_final = "converged"
            log("no new violations -- converged")
            break
        excess = tval[new_m - 1] - TOL
        order = np.argsort(-excess)
        take = new_m[order[:args.add_cap]]
        idxs = (nk + take - 1).astype(np.int32)
        lows = np.full(idxs.size, -INF)
        ups = np.full(idxs.size, TOL)
        try:
            h.changeColsBoundsBySet(idxs.size, idxs, lows, ups)
        except Exception:
            for c in idxs:
                h.changeColBounds(int(c), -INF, TOL)
        W[take] = True
        # After the cold round-1 IPM+crossover solve, switch to warm-started
        # simplex for all cut-generation resolves: fast, gives clean basic duals,
        # and is checkpoint/resume-able on timeout (unlike IPM).
        if current_solver != args.resolve_solver:
            log(f"switching solver {current_solver} -> {args.resolve_solver} "
                f"(warm re-optimization from round-{rnd} basis)")
            if args.resolve_solver == "ipm":
                configure_ipm(h, args)
            else:
                configure_simplex(h, args)
            current_solver = args.resolve_solver

    # ---- dual polish ----
    cert_pre = None
    if status_final == "converged" and args.final_dual_tol > 0:
        try:
            cert_pre = extract_certificate(h, nk, M, ks, cmax, TOL)
            if cert_pre is not None:
                np.savez_compressed(
                    os.path.join(outdir, f"prepolish_{tag}.npz"),
                    m=cert_pre["y_m"], Y=cert_pre["y_num"], denom_pow=48,
                    y=cert_pre["y"], B_float=cert_pre["B_float"],
                    score_lp=cert_pre["score_lp"])
                log(f"pre-polish: B_float={cert_pre['B_float']:.10f} "
                    f"sum|r|={cert_pre['sum_abs_r']:.3e} max|r|={cert_pre['max_abs_r']:.3e}")
            log(f"polish: dual_feasibility_tolerance -> {args.final_dual_tol}")
            h.setOptionValue("dual_feasibility_tolerance", args.final_dual_tol)
            h.setOptionValue("primal_feasibility_tolerance", 1e-8)
            # polish with simplex (clean basic duals) regardless of solve solver
            h.setOptionValue("solver", "simplex")
            arm_limit(h, args.polish_time_limit)   # cumulative timer -> re-arm
            t0 = time.time()
            h.run()
            st = h.getModelStatus()
            log(f"polish: status={h.modelStatusToString(st)} "
                f"score={-h.getObjectiveValue():.10f} time={time.time()-t0:.1f}s")
        except Exception:
            log("polish failed:\n" + traceback.format_exc())

    # ---- final extraction ----
    cert = extract_certificate(h, nk, M, ks, cmax, TOL)
    if cert is not None and cert_pre is not None and cert_pre["B_float"] < cert["B_float"]:
        log(f"polish made B worse ({cert['B_float']:.10f} > {cert_pre['B_float']:.10f}); "
            "keeping pre-polish certificate")
        cert = cert_pre
    if cert is None:
        log("FATAL: no valid duals at end")
        with open(fail_marker, "w") as fh:
            fh.write("no valid duals at end")
        sys.exit(1)

    f = cert["f"]
    E = full_scan(f, ks, M)
    maxE = float(E.max())
    score_f = float(-(f * cmin).sum())
    margin = 1e-7
    alpha = (TOL - margin) / maxE if maxE > TOL - margin else 1.0
    primal_feas = alpha * score_f
    log(f"primal: LP score={cert['score_lp']:.10f} scan maxE={maxE:.10f} "
        f"alpha={alpha:.12f} feasible primal={primal_feas:.10f}")

    r = cert["r"]
    at_bound = np.abs(np.abs(f) - 10.0) < 1e-6
    sum_r_bound = float(np.abs(r[at_bound]).sum())
    sum_r_int = float(np.abs(r[~at_bound]).sum())
    max_r_int = float(np.abs(r[~at_bound]).max()) if (~at_bound).any() else 0.0
    log(f"residuals: max|r|={cert['max_abs_r']:.3e} sum|r|={cert['sum_abs_r']:.6e} "
        f"[at-bound n={int(at_bound.sum())} sum={sum_r_bound:.6e}] "
        f"[interior n={int((~at_bound).sum())} sum={sum_r_int:.6e} max={max_r_int:.3e}]")
    log(f"B_float={cert['B_float']:.10f}  gap B-primal={cert['B_float']-primal_feas:.3e}")

    wlist = np.nonzero(W[1:])[0].astype(np.int64) + 1
    ypath = os.path.join(outdir, f"y_final_{tag}.npz")
    np.savez_compressed(
        ypath,
        m=cert["y_m"], Y=cert["y_num"], denom_pow=48, y=cert["y"],
        working_rows=wlist, f=f, K=K, M=M, TOL=TOL,
        B_float=cert["B_float"], score_lp=cert["score_lp"],
        primal_feasible=primal_feas, alpha=alpha, maxE_scan=maxE)

    report = dict(
        status=status_final, K=K, M=M, TOL=TOL,
        n_working_rows=int(wlist.size), n_cols=int(nk),
        score_lp=cert["score_lp"], primal_feasible=primal_feas,
        maxE_scan=maxE, alpha=alpha,
        B_float=cert["B_float"], sumy=cert["sumy"],
        support=cert["n_support"],
        max_abs_r=cert["max_abs_r"], sum_abs_r=cert["sum_abs_r"],
        sum_r_bound=sum_r_bound, sum_r_interior=sum_r_int, max_r_interior=max_r_int,
        n_at_bound=int(at_bound.sum()),
        rc_crosscheck=cert["rc_crosscheck"], sign=cert["sign"],
        miny_preclip=cert["miny_preclip"],
        round1_solver=args.round1_solver, resolve_solver=args.resolve_solver,
        run_crossover=args.run_crossover,
        y_file=ypath, history=hist)
    with open(os.path.join(outdir, f"report_{tag}.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    log(json.dumps({k: v for k, v in report.items() if k != "history"}))

    with open(done_marker, "w") as fh:
        fh.write(status_final)
    if os.path.exists(resumable_marker):
        os.remove(resumable_marker)
    log("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        try:
            with open("CRASH", "w") as fh:
                fh.write(traceback.format_exc())
        except Exception:
            pass
        sys.exit(1)
