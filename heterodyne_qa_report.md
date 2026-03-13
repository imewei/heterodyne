# Heterodyne QA Report

## Performance Optimization + QA Session — 2026-03-13

| Round | Debug | Review | Commits | Modules | Status |
|-------|-------|--------|---------|---------|--------|
| 1 | 82 mypy errors + 2 gradient-floor fixes | Reverted 3 over-applied gradient changes, fixed scipy import constraint | 3 | core, optimization, utils, viz, data, docs | PASS — 2578 tests, 0 mypy errors |
| 2 | 1 CRITICAL (config mutation in recovery retries) + 2 minor | No review issues | 2 | optimization, __init__ | PASS — 2578 tests, 0 mypy errors |
| 3 | 1 HIGH (CMC warm-start ignores fitted contrast/offset) | Confirmed fix correct | 1 | optimization/cmc | PASS — 2578 tests, 0 mypy errors |
| 4 | 1 MEDIUM (CI key mismatch in MCMC summary) + 1 minor | No review issues | 1 | io | PASS — 2578 tests, 0 mypy errors |
| 5 | 0 findings — physics computation path verified clean | N/A | 0 | — | ALL CLEAR |
