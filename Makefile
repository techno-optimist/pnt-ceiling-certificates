# Makefile -- one-command reproduction of the two certified PNT-board ceilings.
#
#   make verify          both ceilings; prints ALL CEILINGS VERIFIED
#   make verify-K4800    regenerate B(4800) from the archived integer dual and
#                        confirm it equals the theorem's 25-digit bound.
#   make verify-K12000   same for B(12000).
#   make versions        print the exact tool versions in the certified path.
#
# Each verify target runs scripts/recompute_bound.py on the archived integer
# dual (duals/certified_dual_K*.npz) plus its LP report (certs/report_K*.json)
# and certificate (certs/certificate_K*.json), regenerates the exact
# weak-duality bound B by an implementation independent of the original
# certifier, and confirms the pinned 25-digit theorem decimal is a valid, tight
# OUTWARD round-up of that exact rational. Exit status is nonzero on any
# mismatch, and `make` fails loudly.
#
# Requires: python3 with numpy + mpmath (see `make versions`). No network,
# no GPU. highspy/HiGHS is used only in the (separate) LP-solve stage and never
# touches this certified recomputation.

PY ?= python3
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Pinned theorem bounds (the vault values). recompute_bound.py exits 0 iff the
# regenerated exact B is <= the printed decimal (a valid, tight outward round-up).
B4800  := 0.9963688817172828744325041
B12000 := 0.9974876103072528157057480

.PHONY: verify verify-K4800 verify-K12000 versions clean

verify: verify-K4800 verify-K12000
	@echo "ALL CEILINGS VERIFIED."

verify-K4800:
	@echo "== verify-K4800 =="
	$(PY) $(ROOT)scripts/recompute_bound.py $(ROOT)duals/certified_dual_K4800.npz \
	    --cert $(ROOT)certs/certificate_K4800.json \
	    --report $(ROOT)certs/report_K4800.json \
	    --expect-decimal $(B4800) --prec 500 --prec2 800

verify-K12000:
	@echo "== verify-K12000 =="
	$(PY) $(ROOT)scripts/recompute_bound.py $(ROOT)duals/certified_dual_K12000.npz \
	    --cert $(ROOT)certs/certificate_K12000.json \
	    --report $(ROOT)certs/report_K12000.json \
	    --expect-decimal $(B12000) --prec 500 --prec2 800

versions:
	@$(PY) $(ROOT)scripts/print_versions.py

clean:
	@echo "(nothing to clean; verification is read-only)"
