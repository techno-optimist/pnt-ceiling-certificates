#!/usr/bin/env python3
"""Figure for the S*(K) ceiling note: certified ceilings vs the board
trajectory (panel a) and the c-units view (panel b).

Display-only; every number is quoted from the verified artifacts (see the
note's Appendix). No certified content is computed here.
"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- panel (a)
# Board trajectory (submitted scores, tolerance tau = 1.0001 semantics),
# retrieved from the live leaderboard 2026-07-06: (reach = max key, score).
board = [
    (16000, 0.9963196933439522, "MAOJIASONG"),
    (24000, 0.9965177307112617, "CHRONOS"),
    (32001, 0.9971452043881762, "Agent-Knowledge-Cycle"),
    (47999, 0.9973457049300725, "JSAgent"),
    (63998, 0.9973964360068646, "JSAgent (leader)"),
]
# Certified ceilings (exact, rounded up in the last displayed digit).
ceilings = [(4800, 0.9963688817172829), (12000, 0.9974876103072528)]
# Measured cardinality curve at K = 12000 (floats, NOT certificates).
card = [(2000, 0.9964671727633392), (3000, 0.9969336739187579),
        (4000, 0.9971949170287637), (11999, 0.9974600430839198)]

# ---------------------------------------------------------------- panel (b)
# Honest c-units ladder, c := (1 - S_honest) ln(10 reach) at RHS = 1.0,
# as published by the agents (threads 244 / 251).
akc = [(16000, 0.0392), (24000, 0.0370), (32000, 0.0362)]
js = [(48000, 0.0360), (64000, 0.0361)]
# Certified floors (this note), valid for tau = 1.0001 hence for RHS = 1.0.
floors = [(4800, 0.0391396652), (12000, 0.0293830181)]
# Our measured best 2000-key support at reach 12000 (tolerance semantics).
ours_c = (12000, 0.0413173)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0))

# ----- panel (a)
xs = [b[0] for b in board]
ys = [b[1] for b in board]
ax1.plot(xs, ys, "o-", color="#1f77b4", ms=5, lw=1.2, zorder=3,
         label="board submissions (tolerance semantics)")
for K, B in ceilings:
    ax1.plot([K], [B], marker="v", color="#d62728", ms=9, zorder=4)
    ax1.annotate(f"$B({K})$", (K, B), textcoords="offset points",
                 xytext=(-34, 2), fontsize=9, color="#d62728")
# the four measured supports all live at reach K = 12000
cy = [c[1] for c in card]
ax1.plot([12000] * len(card), cy, "s", color="#2ca02c", ms=5, zorder=3,
         mfc="none", label="measured supports at $K=12000$ (float)")
for (k, s), lab in zip(card, ["2000 keys", "3000 keys", "4000 keys",
                              "11999 (full)"]):
    ax1.annotate(lab, (12000, s), textcoords="offset points",
                 xytext=(7, -3), fontsize=7.5, color="#2ca02c")
ax1.axhline(1.0, color="0.6", lw=0.8, ls=":")
ax1.annotate("asymptotic value 1 (full Möbius)", (13500, 1.0),
             textcoords="offset points", xytext=(0, -10), fontsize=8,
             color="0.35")
ax1.set_xscale("log")
ax1.set_xticks([4800, 12000, 24000, 48000])
ax1.set_xticklabels(["4800", "12000", "24000", "48000"])
ax1.set_xlabel("reach $K$ (max key)")
ax1.set_ylabel("score")
ax1.set_ylim(0.99615, 1.0007)
ax1.legend(fontsize=8, loc="upper left")
ax1.set_title("(a) certified ceilings vs the board trajectory", fontsize=10)

# ----- panel (b)
ax2.plot([p[0] for p in akc], [p[1] for p in akc], "o-", color="#1f77b4",
         ms=5, lw=1.2, label="honest ladder (Agent-Knowledge-Cycle)")
ax2.plot([p[0] for p in js], [p[1] for p in js], "D-", color="#17becf",
         ms=5, lw=1.2, label="plateau (JSAgent)")
for K, c in floors:
    ax2.plot([K], [c], marker="^", color="#d62728", ms=9, zorder=4)
    ax2.annotate(f"$c^*({K})$", (K, c), textcoords="offset points",
                 xytext=(-52, -3), fontsize=9, color="#d62728")
ax2.plot([ours_c[0]], [ours_c[1]], "s", color="#2ca02c", ms=6, mfc="none")
ax2.annotate("best 2000-key at 12000 (measured)", ours_c,
             textcoords="offset points", xytext=(8, -12), fontsize=7.5,
             color="#2ca02c")
ax2.axhline(0.036, color="0.4", lw=0.9, ls="--")
ax2.annotate("0.036 (open-question line)", (4900, 0.036),
             textcoords="offset points", xytext=(0, 4), fontsize=8,
             color="0.35")
ax2.set_xscale("log")
ax2.set_xticks([4800, 12000, 24000, 48000])
ax2.set_xticklabels(["4800", "12000", "24000", "48000"])
ax2.set_xlabel("reach (max key)")
ax2.set_ylabel(r"$c = (1-S)\,\ln(10\cdot\mathrm{reach})$")
ax2.set_ylim(0.027, 0.0445)
ax2.legend(fontsize=8, loc="lower right")
ax2.set_title("(b) structure constant: floors, ladder, open question",
              fontsize=10)

fig.tight_layout()
fig.savefig("fig_pnt_ceilings.pdf")
fig.savefig("fig_pnt_ceilings.png", dpi=180)
print("wrote fig_pnt_ceilings.pdf / .png")
