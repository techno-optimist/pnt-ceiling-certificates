#!/usr/bin/env python
"""Rehearse the dual-polish on the converged mid (K=4800) problem:
re-solve with the final working set, then re-run with tightened dual tolerance
(fresh INVERT, unperturbed costs) and measure the residual profile of the duals.
Also saves the optimal basis for the sparse-LU lambda route.
"""
import numpy as np, highspy, time, sys, json
sys.path.insert(0, ".")  # solve-stage helper; expects solve_sstar.py on path
from solve_sstar import build_lp, extract_certificate, log

INF = highspy.kHighsInf
K = 4800; M = 10 * K; TOL = 1.0001
d = np.load("y_final_K4800.npz")
W = d["working_rows"].astype(np.int64)
ks = np.arange(2, K + 1, dtype=np.int64)
nk = ks.size
cmin = np.log(ks.astype(np.float64)) / ks
cmax = -cmin

t_upper = np.full(M, INF)
t_upper[W - 1] = TOL
lp = build_lp(K, M, TOL, t_upper, ks, cmin)
h = highspy.Highs()
h.setOptionValue("threads", 8)
h.setOptionValue("solver", "simplex")
h.setOptionValue("output_flag", True)
h.passModel(lp)
del lp
t0 = time.time()
h.run()
log(f"solve1: {h.modelStatusToString(h.getModelStatus())} "
    f"score={-h.getObjectiveValue():.10f} time={time.time()-t0:.1f}s")
cert = extract_certificate(h, nk, M, ks, cmax, TOL)
log(f"solve1 cert: B={cert['B_float']:.10f} sum|r|={cert['sum_abs_r']:.3e} "
    f"max|r|={cert['max_abs_r']:.3e} support={cert['n_support']} "
    f"miny={cert['miny_preclip']:.2e}")

for tol in (1e-9, 1e-10):
    h.setOptionValue("dual_feasibility_tolerance", tol)
    h.setOptionValue("primal_feasibility_tolerance", 1e-8)
    t0 = time.time()
    h.run()
    log(f"polish tol={tol}: {h.modelStatusToString(h.getModelStatus())} "
        f"score={-h.getObjectiveValue():.10f} time={time.time()-t0:.1f}s")
    cert = extract_certificate(h, nk, M, ks, cmax, TOL)
    log(f"polish tol={tol} cert: B={cert['B_float']:.10f} sum|r|={cert['sum_abs_r']:.3e} "
        f"max|r|={cert['max_abs_r']:.3e} support={cert['n_support']} "
        f"miny={cert['miny_preclip']:.2e}")
    np.savez_compressed(f"y_polish{tol:.0e}_K4800.npz",
                        m=cert["y_m"], Y=cert["y_num"], denom_pow=48, y=cert["y"],
                        f=cert["f"], working_rows=W, K=K, M=M, TOL=TOL,
                        B_float=cert["B_float"])

h.writeBasis("basis_K4800.txt")
log("basis written; DONE")
