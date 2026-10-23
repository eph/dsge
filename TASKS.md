# Project tasks

This is the repository's task list. Edit it directly and commit task updates with
related changes; no tracker service or CLI is needed. Keep task IDs stable, add
new tasks with an unused `task-NNN` ID, and mark completed entries `[x]` with a
completion date and a short resolution. Keep completed entries for context.

## Migration provenance

On 2026-10-02, the 10 open tasks below were copied from `.beads/issues.jsonl` at
commit `0cfeca45eafdaefb420fb78ddba5fb1c6f319fe9`. Their status, descriptions,
priorities, and recorded metadata are preserved as found. Priority numbers and
issue types retain their original values. Dated audit notes are separate from
those original records.

The [inactive archive](archive/beads/README.md) contains all 27 original
records (10 open and 17 closed), including closed-task history and dependencies,
plus recovery instructions. The archive is a snapshot; maintain this file for
ongoing work.

## Audit on 2026-10-02

All 10 tasks remain open after checking GitHub `master` at
`0cfeca45eafdaefb420fb78ddba5fb1c6f319fe9` and the separate, unmerged
`codex/security-cloud-fixes` branch at
`5eff5e011f59cee81b278d053702254d4d8ec03b`. Both remote heads were verified on the
audit date. No task is verified resolved; no task is left unclassified. Bug
reproductions and code/API findings for missing features are distinguished below.
See the [full audit and reproduction commands](archive/beads/audit-2026-10-02.md).

## Tasks

### [ ] dsge-0qi — Remove hardcoded local include paths from generated C++ CMake template

- Status: open
- Priority: 2
- Type: task
- Owner: ed.herbst@gmail.com
- Created by: Ed Herbst
- Created: 2026-03-13T14:40:18.191886891-04:00
- Updated: 2026-03-13T14:40:18.191886891-04:00
- Labels: cpp, portability, post-release

translate_cpp currently writes a CMakeLists.txt with author-specific absolute include directories for Eigen, Stan Math, Boost, and Sundials. Replace those with configurable discovery or documented inputs so generated exports are portable.

**Audit 2026-10-02:** Still reproduces on both branches. A fresh FHP C++ export writes the same four machine-specific Eigen, Stan Math, Boost, and Sundials include directories to `CMakeLists.txt`.

### [ ] dsge-5fc — Fix anticipated_impulse_response for anticipated_h=0

- Status: open
- Priority: 2
- Type: bug
- Owner: ed.herbst@gmail.com
- Created by: Ed Herbst
- Created: 2026-03-13T16:42:06.124081775-04:00
- Updated: 2026-03-13T16:42:06.124081775-04:00

LinearDSGEModel.anticipated_impulse_response raises a shape mismatch when called with anticipated_h=0 because RR[:, neps:] is empty in that branch. anticipated_h>=1 works and matches the IRFOC anticipated-shock mapping numerically. The zero-horizon case should fall back to the standard contemporaneous impulse response instead of failing.

**Audit 2026-10-02:** Still reproduces on both branches. A one-state AR(1) model has a valid ordinary IRF, but `anticipated_h=0` raises `ValueError: shapes (1,0) and (1,1) not aligned`.

### [ ] dsge-847 — Make DGS external helper path portable

- Status: open
- Priority: 2
- Type: bug
- Created by: ed
- Created: 2026-09-03T14:15:58.822106-04:00
- Updated: 2026-09-03T14:15:58.822106-04:00

The full test suite on macOS fails in TestExternal.test_DGS because the DGS example references /home/eherbst/Dropbox/code/dsge/dsge/examples/DGS/helper_func.py. Reproduced on unchanged base revision 11599b9. Resolve the helper relative to the packaged YAML/resource location so the example compiles on other machines.

**Audit 2026-10-02:** Still reproduces on master. The security branch partially addresses the test by explicitly supplying the bundled helper registry, so its DGS test passes. However, the shipped YAML still names the stale absolute path, and file-based compilation with `allow_external_code=True` raises `FileNotFoundError` on security too. Keep open for the remaining portable example/file-path workflow.

### [ ] dsge-848 — Make Python-Numba filter parity test numerically portable

- Status: open
- Priority: 2
- Type: bug
- Created by: ed
- Created: 2026-09-03T14:15:58.853175-04:00
- Updated: 2026-09-03T14:15:58.853175-04:00

The full test suite on macOS fails test_kalman_filter_numba_matches_python because it requires exact bitwise equality. Python and Numba log likelihoods differ by 7.73070496506989e-12 at values near -4045.37. Reproduced on unchanged base revision 11599b9. Use a justified floating-point tolerance or identify and remove the operation-order difference if exact identity is intended.

**Audit 2026-10-02:** Still reproduces on both branches. The existing Kalman Python/Numba parity test fails because the absolute difference is `7.73070496506989e-12` and the assertion requires exactly zero. Keep open; no tolerance change was made.

### [ ] dsge-850 — Preserve FHP output precision at extremely uneven planning horizons

- Status: open
- Priority: 2
- Type: bug
- Created by: ed
- Created: 2026-09-03T15:30:50.243247-04:00
- Updated: 2026-09-03T15:30:50.243247-04:00

The FHP reduced-state reconstruction can lose output precision when own-horizon projection combines a moderate household horizon with an extremely long pricing horizon. This predates exponential-cost YAML support and is separate from cost selection.

Reproduction with `dsge/examples/fhp/nk_endogenous_exponential_costs.yaml`: set both `k_max` values to 370, load the model, use the calibrated growth 0.31, and set the reduced state to `[0.0025]`. Native `choose_regime` correctly returns `(28, 369)`. However, evaluating `ZZ @ x + DD` from `get_mats` can produce output near -508640.7 instead of the exact triangular GE value 0.6659686781, while inflation is approximately 1.6197794421e21.

The exact own-horizon allocation is `z = d*b + MH*common(h-1) + MF*common(f-1)`, where `common(k) = d*sum(M**j*b, j=0..k)`. In this NK model the output row has no dependence on the firm's continuation, so enormous inflation should not contaminate the household output calculation.

Investigate the reachable-subspace reconstruction `G = Q @ pinv(Qs)` in `FHPRepAgent.compile_endogenous_horizon_switching`, including scaling and structurally sparse reconstruction. Preserve accurate allocations over the moderate plotted range and add a regression test for a strongly uneven, long-horizon profile. The new exponential-cost feature tests allocations through horizon 59 and separately confirms the native stopping profile at the larger cap.

**Audit 2026-10-02:** Still reproduces on both branches. With both caps set to 370 and state `[0.0025]`, the chosen regime is `(28, 369)`; reconstructed output is `-508640.697359835` versus `0.6659686781019037` from an independent triangular recurrence. The existing moderate-horizon test passes and does not cover this extreme case.

### [ ] dsge-gwm — Support generic IRFOC prehistory for arbitrary lagged endogenous variables

- Status: open
- Priority: 2
- Type: task
- Owner: ed.herbst@gmail.com
- Created by: Ed Herbst
- Created: 2026-03-13T15:49:57.621646707-04:00
- Updated: 2026-03-13T15:49:57.621646707-04:00

IRFOC now handles explicit lag notation correctly when the model/baseline exposes an exact lag alias column such as ilag = i(-1). In-window lagged endogenous variables like pi(-1), y(-1), and pi(-2) work correctly, but at the left boundary t=0 explicit lags still fall back to zero unless matching alias columns or other presample history are provided. Follow up by deciding whether IRFOC should accept a generic presample history vector for arbitrary endogenous lags, infer it from baseline metadata, or document the alias-column requirement more explicitly.

**Audit 2026-10-02:** Still reproduces on both branches. For a baseline shifted by five periods with known prior inflation `-0.04676688917187356` and no inflation-lag alias, `i = 0.5*pi(-1)` produces initial `i` near zero instead of `-0.02338344458593678`. Existing alias-based history tests pass; generic presample input remains absent.

### [ ] dsge-845 — IRFOC: add horizon padding helper for forward-looking accuracy

- Status: open
- Priority: 3
- Type: task
- Owner: ed.herbst@gmail.com
- Created by: Ed Herbst
- Created: 2026-03-13T20:33:13.269506714-04:00
- Updated: 2026-03-13T20:33:13.269506714-04:00

IRFOC counterfactuals can require a longer solve horizon than the plotted/compared IRF horizon (e.g. SW needs T~120 to match 0..60 to ~1e-5). Add a helper to extend/pad a baseline IRF (zero padding or user-provided tail) and return only the requested compare window; document recommended practice.

**Audit 2026-10-02:** Still open on both branches (helper absent by code/API inspection). IRFOC takes its solve horizon from the supplied baseline rows and returns that same window; no padding/tail helper or separate comparison-window interface is present. This is a missing-feature finding, not a failing numerical test.

### [ ] dsge-846 — Add owner-specific forecasts for own-horizon projection

- Status: open
- Priority: 3
- Type: feature
- Created by: ed
- Created: 2026-09-03T14:15:58.791298-04:00
- Updated: 2026-09-03T14:15:58.791298-04:00

The new own_horizon_projection mode solves current allocations and endogenous horizon choices, but rejects declarations.expectations > 0 because an off-diagonal horizon profile does not imply a unique aggregate subjective forecast path. Add owner-specific forecast observables (for example household and pricing forecasts) with explicit countdown semantics and tests that distinguish the two diagonal perceived paths. Preserve the current explicit rejection until that interface is defined.

**Audit 2026-10-02:** Still open on both branches (feature explicitly unsupported). Setting `declarations.expectations` to 1 in the own-horizon example raises the documented error about different row owners. The existing rejection test passes; owner-specific forecast observables have not been implemented.

### [ ] dsge-849 — Make documented unittest discovery preserve package imports

- Status: open
- Priority: 3
- Type: bug
- Created by: ed
- Created: 2026-09-03T14:19:14.512746-04:00
- Updated: 2026-09-03T14:19:14.512746-04:00

The documented command python -m unittest discover -s dsge/tests -p test_*.py imports test_fhp as a top-level module. Its test_irf_fhp and test_irf_re methods use relative imports and fail with ImportError: attempted relative import with no known parent package. Update the documented invocation with an appropriate top-level directory or make test imports portable. The same tests pass under pytest; this is unrelated to own-horizon projection.

**Audit 2026-10-02:** Still reproduces on both branches. Documented discovery, narrowed to `test_fhp.py` / `test_irf_fhp`, raises `ImportError: attempted relative import with no known parent package`. Adding `-t .` passes the same test as a control. The documented command and relative imports remain unchanged.

### [ ] dsge-y4d — Add IRFOC reachability diagnostics for off-span targets

- Status: open
- Priority: 3
- Type: task
- Owner: ed.herbst@gmail.com
- Created by: Ed Herbst
- Created: 2026-03-13T16:30:14.761240679-04:00
- Updated: 2026-03-13T16:30:14.761240679-04:00

Expose a small diagnostic for comparing a target path to the finite-horizon instrument span used by IRFOC. Useful outputs would be a least-squares fitted shock path, projection residuals by variable/horizon, and a quick way to tell whether an exact parameter-change counterfactual is reachable under the chosen instrument set and baseline dynamics.

**Audit 2026-10-02:** Still open on both branches (diagnostic absent by code/API inspection). `IRFOCResult.residuals` contains policy-rule residuals, not target-path projection errors. There is no public target/reachability diagnostic returning fitted shocks and projection residuals; the internal least-squares fallback does not provide that feature.
